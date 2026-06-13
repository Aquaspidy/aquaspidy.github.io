import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import datetime
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

# ====================================================================
# CONFIGURATION - All magic numbers in one place
# ====================================================================

@dataclass
class AppConfig:
    """Central configuration for all tunable parameters"""
    # Video settings
    frame_width: int = 960
    frame_height: int = 540
    frame_skip: int = 3  # Process every Nth frame (1=all frames, 2=half)
    
    # Camera settings
    camera_warmup_frames: int = 30  # Frames to discard on startup
    camera_id: int = 0
    
    # Detection settings
    detection_interval: int = 2  # Run YOLO every N frames
    min_tracking_confidence: float = 0.5
    
    # Team color detection
    color_coverage_threshold: float = 0.05  # Min % of jersey area for team color
    color_dominance_ratio: float = 2.0  # Must be 2x opponent color
    identity_lock_frames: int = 15  # Frames needed to lock player identity
    memory_fallback_distance: int = 70  # Pixels for spatial memory fallback
    memory_buffer_size: int = 50
    
    # Jersey cropping (relative to bounding box)
    jersey_top_pct: float = 0.15  # Skip top 15% (head/hair)
    jersey_bottom_pct: float = 0.60  # Bottom 40% (shorts)
    jersey_side_trim_pct: float = 0.20  # Trim 20% from sides
    
    # Optical flow
    flow_feature_max_corners: int = 100
    flow_feature_quality: float = 0.3
    flow_feature_min_distance: int = 7
    flow_win_size: tuple = (15, 15)
    flow_max_level: int = 2
    
    # Possession momentum
    momentum_change_strength: int = 5
    momentum_decay: float = 0.99
    momentum_threshold: int = 75
    momentum_max: int = 100
    movement_threshold: float = 1.5  # Pixels per frame to register intent
    
    # Cramping detection
    cramped_spread_threshold: int = 400  # Pixels
    rim_region_pct: float = 0.30  # Left/right 30% of screen
    middle_region_pct: tuple = (0.30, 0.60)  # Middle 30-60% of screen
    
    # PnR detection
    pnr_min_distance: int = 20
    pnr_max_distance: int = 80
    pnr_evaluation_frames: int = 45
    pnr_success_threshold: int = 10  # qSQ improvement needed
    
    # Mismatch detection
    mismatch_min_dist: int = 15
    mismatch_max_dist: int = 75
    mismatch_height_ratio_min: float = 1.25
    mismatch_height_ratio_max: float = 1.6
    mismatch_vertical_sep: int = 15
    
    # Defensive shell
    compact_shell_area_ratio: float = 0.12  # <12% of frame = compact
    
    # Free throw detection
    ft_min_speed: float = 1.5  # Movement threshold
    ft_lane_width: int = 150  # Pixels from center
    ft_shooter_distance: int = 150  # Min distance from paint
    
    # Report settings
    qsq_min: int = 10
    qsq_max: int = 99
    
    # Model path
    model_name: str = 'yolov8n.pt'
    
    # Timeout report thresholds
    pace_elite_threshold: float = 55.0
    pace_avg_threshold: float = 25.0
    mismatch_action_threshold: int = 3


# ====================================================================
# LOGGING SETUP
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================================================================
# LIGHTING NORMALIZATION
# ====================================================================

def normalize_lighting(frame):
    """Fix colors so team detection works in any lighting"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    normalized = cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)
    return normalized

# ====================================================================
# ANALYTICS ENGINE (Improved)
# ====================================================================

class TimeoutScout:
    """Generates timeout reports from accumulated analytics"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.pnr_log: List[float] = []
        self.mismatch_count: int = 0
        self.pace_log: List[float] = []
        self.sq_log: List[float] = []
        self.spacing_log: List[int] = []
        self.cramp_events: Dict[str, int] = {
            'RIM_CRAMP_GUARDS': 0,
            'RIM_CRAMP': 0,
            'MIDDLE_STAGNANT': 0
        }
    
    def log_pnr(self, start_qsq: float, end_qsq: float) -> None:
        self.pnr_log.append(end_qsq - start_qsq)
    
    def log_mismatch(self) -> None:
        self.mismatch_count += 1
    
    def log_pace(self, pace: float) -> None:
        if pace > 5:
            self.pace_log.append(pace)
    
    def log_shot_quality(self, qsq_val: float) -> None:
        self.sq_log.append(qsq_val)
    
    def log_spacing(self, width: int) -> None:
        self.spacing_log.append(width)
    
    def log_cramp(self, cramp_type: str) -> None:
        if cramp_type in self.cramp_events:
            self.cramp_events[cramp_type] += 1
    
    def _generate_pace_text(self, avg_pace: float) -> str:
        if avg_pace > self.config.pace_elite_threshold:
            return f"Pace is ELITE (Rating: {int(avg_pace)}). Excellent transition speed."
        elif avg_pace > self.config.pace_avg_threshold:
            return f"Pace is AVERAGE (Rating: {int(avg_pace)}). Good half-court flow, look for more hard cuts."
        return f"Pace is STAGNANT (Rating: {int(avg_pace)}). We are standing and watching the ball."
    
    def _generate_mismatch_text(self) -> str:
        if self.mismatch_count >= self.config.mismatch_action_threshold:
            return f"Forced {self.mismatch_count} size mismatches. Keep hunting switches and dump it inside!"
        elif self.mismatch_count > 0:
            return "Occasionally getting height advantages on switches. Look for the high-low pass."
        return "Defense is fighting through screens well; no obvious size advantages created."
    
    def _generate_pnr_text(self) -> str:
        if not self.pnr_log:
            return "No Pick & Rolls tracked yet."
        
        avg_space = sum(self.pnr_log[-5:]) / min(5, len(self.pnr_log))
        if avg_space > 15:
            return f"PnR is highly effective (+{int(avg_space)}% separation). Keep spamming it."
        elif avg_space > 0:
            return f"PnR is moderately effective. Roll harder to the rim."
        return "PnR is getting blown up. Space the floor or slip the screen early."
    
    def _generate_shot_quality_text(self) -> str:
        if not self.sq_log:
            return "Not enough offensive data to determine Shot Quality."
        
        expected_fgs = [int(20 + (sq * 0.45)) for sq in self.sq_log]
        buckets = {
            "HIGH": [fg for fg in expected_fgs if fg >= 50],
            "MEDIUM": [fg for fg in expected_fgs if 35 <= fg < 50],
            "LOW": [fg for fg in expected_fgs if fg < 35]
        }
        
        total = len(expected_fgs)
        summary_parts = []
        for name, values in buckets.items():
            if values:
                percent = int((len(values) / total) * 100)
                avg_make = int(sum(values) / len(values))
                summary_parts.append(f"{percent}% of shots are {name} quality (~{avg_make}% expected FG)")
        
        return "SHOT QUALITY BREAKDOWN:\n" + "\n".join(summary_parts)
    
    def _generate_spacing_text(self) -> Tuple[str, str]:
        spread_text = ""
        cramp_text = "Spacing has generally been adequate."
        
        if self.spacing_log:
            avg_width = int(sum(self.spacing_log) / len(self.spacing_log))
            spread_text = f"OFFENSIVE SPREAD: Average width is {avg_width} pixels. "
        
        total_cramps = sum(self.cramp_events.values())
        if total_cramps > 0:
            most_common = max(self.cramp_events, key=self.cramp_events.get)
            if most_common == 'RIM_CRAMP_GUARDS' and self.cramp_events['RIM_CRAMP_GUARDS'] >= 2:
                cramp_text = (f"CRAMPED UNDER RIM ({self.cramp_events['RIM_CRAMP_GUARDS']} times). "
                              "Our shorter guards are getting stuck in the paint. Clear them out to the corners!")
            elif most_common == 'RIM_CRAMP' and self.cramp_events['RIM_CRAMP'] >= 2:
                cramp_text = (f"CRAMPED UNDER RIM ({self.cramp_events['RIM_CRAMP']} times). "
                              "The paint is entirely clogged. Bigs need to space out to the perimeter.")
            elif most_common == 'MIDDLE_STAGNANT' and self.cramp_events['MIDDLE_STAGNANT'] >= 2:
                cramp_text = (f"STUCK IN THE MIDDLE ({self.cramp_events['MIDDLE_STAGNANT']} times). "
                              "We are clustered at the top of the key. DRIVE MORE!")
        
        return spread_text, cramp_text
    
    def generate_timeout_report(self) -> str:
        """Generate and save timeout report, returns filename"""
        current_time = datetime.datetime.now().strftime('%H%M%S')
        filename = f"TIMEOUT_REPORT_{current_time}.txt"
        
        avg_pace = sum(self.pace_log) / len(self.pace_log) if self.pace_log else 0
        
        pace_text = self._generate_pace_text(avg_pace)
        mismatch_text = self._generate_mismatch_text()
        pnr_text = self._generate_pnr_text()
        sq_text = self._generate_shot_quality_text()
        spread_text, cramp_text = self._generate_spacing_text()
        
        report_content = (
            f"--- TIMEOUT REPORT ---\n\n"
            f"{pace_text}\n\n"
            f"{mismatch_text}\n\n"
            f"{pnr_text}\n\n"
            f"{sq_text}\n\n"
            f"{spread_text}{cramp_text}"
        )
        
        with open(filename, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved: {filename}")
        
        # Reset logs
        self.pnr_log.clear()
        self.mismatch_count = 0
        self.sq_log.clear()
        self.spacing_log.clear()
        self.cramp_events = {k: 0 for k in self.cramp_events}
        
        return filename


# ====================================================================
# COLOR DETECTION (Improved)
# ====================================================================

class TeamColorDetector:
    """Handles team color detection - NOW WITH LEARN BY CLICKING"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        
        # LEARNING MODE STORAGE
        self.home_samples = []  # Stores HSV values from clicks
        self.away_samples = []
        self.home_learned = False
        self.away_learned = False
        self.learning_active = False
        self.learning_team = None  # 'home' or 'away'
    
    def start_learning(self, team):
        """Call this to start learning mode"""
        self.learning_active = True
        self.learning_team = team
        if team == 'home':
            self.home_samples = []
        else:
            self.away_samples = []
        print(f"Learning mode: Click on {team} player's jersey")
    
    def add_sample_from_click(self, crop):
        """Call this when user clicks on a player - pass the jersey crop"""
        if not self.learning_active or crop is None or crop.size == 0:
            return False, 0
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv, axis=(0, 1))
        
        if self.learning_team == 'home':
            self.home_samples.append(avg_hsv)
            count = len(self.home_samples)
            if count >= 5:
                self.home_learned = True
                self.learning_active = False
                print(f"Home jersey learned! ({count} samples)")
            return True, count
        else:
            self.away_samples.append(avg_hsv)
            count = len(self.away_samples)
            if count >= 5:
                self.away_learned = True
                self.learning_active = False
                print(f"Away jersey learned! ({count} samples)")
            return True, count
    
    def detect_team(self, player_crop: np.ndarray, t1_range: Tuple, t2_range: Tuple) -> str:
        """Detect team - uses learned samples if available, otherwise falls back to HSV ranges"""
        
        # PRIORITY 1: Use learned samples (if available)
        if self.home_learned and player_crop.size > 0:
            hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
            crop_avg = np.mean(hsv, axis=(0, 1))
            
            # Check against home samples
            for sample in self.home_samples:
                if np.linalg.norm(crop_avg - sample) < 50:
                    return "Westy"
            
            # Check against away samples
            for sample in self.away_samples:
                if np.linalg.norm(crop_avg - sample) < 50:
                    return "Opponent"
        
        # PRIORITY 2: Fall back to HSV range detection (your original working code)
        if player_crop.size == 0:
            return "Unknown"
        
        lab = cv2.cvtColor(player_crop, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        area = player_crop.shape[0] * player_crop.shape[1]
        
        mask1 = cv2.inRange(hsv, np.array(t1_range[0]), np.array(t1_range[1]))
        mask2 = cv2.inRange(hsv, np.array(t2_range[0]), np.array(t2_range[1]))
        
        score1 = np.count_nonzero(mask1) / area
        score2 = np.count_nonzero(mask2) / area
        
        if score1 > self.config.color_coverage_threshold and score1 > score2 * self.config.color_dominance_ratio:
            return "Westy"
        if score2 > self.config.color_coverage_threshold and score2 > score1 * self.config.color_dominance_ratio:
            return "Opponent"
        
        return "Unknown"


# ====================================================================
# PLAYER TRACKING STATE
# ====================================================================

@dataclass
class PlayerData:
    """Data for a single tracked player"""
    id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=20))
    team_votes: Dict[str, int] = field(default_factory=lambda: {"Westy": 0, "Opponent": 0})
    team_locked: Optional[str] = None
    last_qsq: float = 50.0


class PlayerTracker:
    """Manages player tracking state and identity"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.players: Dict[int, PlayerData] = {}
        self.recent_memory: List[Tuple[int, int, str]] = []  # x, y, team
        self.color_detector = TeamColorDetector(config)
    
    def update_team_vote(self, player_id: int, team_guess: str) -> None:
        """Update voting for player's team identity"""
        if player_id not in self.players:
            self.players[player_id] = PlayerData(id=player_id)
        
        if team_guess != "Unknown":
            self.players[player_id].team_votes[team_guess] += 1
    
    def get_locked_team(self, player_id: int) -> Optional[str]:
        """Get locked team identity if enough votes accumulated"""
        player = self.players.get(player_id)
        if not player or player.team_locked:
            return player.team_locked if player else None
        
        # Lock identity after threshold frames
        if player.team_votes["Westy"] > self.config.identity_lock_frames:
            player.team_locked = "Westy"
        elif player.team_votes["Opponent"] > self.config.identity_lock_frames:
            player.team_locked = "Opponent"
        
        return player.team_locked
    
    def get_team_with_memory_fallback(self, player_id: int, cx: int, cy: int) -> str:
        """Get team with spatial memory fallback when identity unknown"""
        team = self.get_locked_team(player_id)
        
        if team:
            return team
        
        # Fallback to nearest neighbor in memory
        for px, py, mem_team in reversed(self.recent_memory):
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < self.config.memory_fallback_distance:
                return mem_team
        
        return "Unknown"
    
    def update_memory(self, cx: int, cy: int, team: str) -> None:
        """Update spatial memory for fallback"""
        if team != "Unknown":
            self.recent_memory.append((cx, cy, team))
            if len(self.recent_memory) > self.config.memory_buffer_size:
                self.recent_memory.pop(0)
    
    def update_position(self, player_id: int, cx: int, cy: int) -> None:
        """Update player's position history"""
        if player_id not in self.players:
            self.players[player_id] = PlayerData(id=player_id)
        self.players[player_id].positions.append((cx, cy))
    
    def get_recent_positions(self, player_id: int) -> List[Tuple[int, int]]:
        """Get recent positions for a player"""
        player = self.players.get(player_id)
        return list(player.positions) if player else []
    
    def get_player(self, player_id: int) -> Optional[PlayerData]:
        return self.players.get(player_id)


# ====================================================================
# COURT CALIBRATION
# ====================================================================

class CourtCalibrator:
    """Handles court polygon calibration via mouse clicks"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.calibration_points: List[List[int]] = []
        self.court_polygon: Optional[np.ndarray] = None
    
    def calibrate(self) -> bool:
        """Run calibration routine, returns True if successful"""
        # Try camera 0 first, then 1 if fails
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera 0 failed, trying camera 1")
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: Could not open any camera")
            return
        if not cap.isOpened():
            logger.error("Could not open camera for calibration")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error("Could not read frame for calibration")
            return False
        
        self.calibration_points = []
        frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
        
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.calibration_points.append([x, y])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                if len(self.calibration_points) > 1:
                    cv2.line(frame, tuple(self.calibration_points[-2]), 
                            tuple(self.calibration_points[-1]), (255, 0, 0), 2)
                cv2.imshow("DEFINE ACTIVE COURT", frame)
        
        cv2.imshow("DEFINE ACTIVE COURT", frame)
        cv2.setMouseCallback("DEFINE ACTIVE COURT", click_event)
        print("CLICK 4 CORNERS (Clockwise: Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
        
        while len(self.calibration_points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("DEFINE ACTIVE COURT")
        
        if len(self.calibration_points) == 4:
            self.court_polygon = np.array(self.calibration_points, dtype=np.int32)
            logger.info("Court calibrated successfully")
            return True
        
        logger.warning("Calibration incomplete, using default full-frame court")
        self.court_polygon = np.array([[0, 0], [self.config.frame_width, 0],
                                       [self.config.frame_width, self.config.frame_height],
                                       [0, self.config.frame_height]], dtype=np.int32)
        return False
    
    def is_inside_court(self, x: int, y: int) -> bool:
        """Check if point is inside calibrated court area"""
        if self.court_polygon is None:
            return True
        return cv2.pointPolygonTest(self.court_polygon, (int(x), int(y)), False) >= 0


# ====================================================================
# OPTICAL FLOW TRACKER
# ====================================================================

class OpticalFlowTracker:
    """Handles camera motion compensation via optical flow"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.prev_gray = None
        self.p0 = None
        self.feature_params = dict(
            maxCorners=config.flow_feature_max_corners,
            qualityLevel=config.flow_feature_quality,
            minDistance=config.flow_feature_min_distance,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=config.flow_win_size,
            maxLevel=config.flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def update(self, gray: np.ndarray, frame_count: int, y_cutoff: int) -> Tuple[float, float]:
        """Update optical flow and return camera movement (dx, dy)"""
        cam_dx, cam_dy = 0.0, 0.0
        
        if self.prev_gray is not None and self.p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.p0, None, **self.lk_params
            )
            if p1 is not None and len(p1[st == 1]) > 0:
                cam_dx = np.median(p1[st == 1][:, 0] - self.p0[st == 1][:, 0])
                cam_dy = np.median(p1[st == 1][:, 1] - self.p0[st == 1][:, 1])
        
        # Update tracking points periodically
        if frame_count % 30 == 0 or self.p0 is None:
            mask = np.zeros_like(gray)
            mask[y_cutoff:, :] = 255
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        self.prev_gray = gray.copy()
        return cam_dx, cam_dy


# ====================================================================
# POSSESSION ENGINE
# ====================================================================

class PossessionEngine:
    """Determines which team has possession based on movement momentum"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.momentum = 0.0
        self.current_possession = "Unknown"
        self.westy_attacks_right = False
    
    def update(self, player_movements: List[float], westy_attacks_right: bool) -> None:
        """Update possession based on player movements"""
        if len(player_movements) < 3:
            self.momentum *= self.config.momentum_decay
            return
        
        team_intent = np.median(player_movements)
        
        if abs(team_intent) > self.config.movement_threshold:
            if not westy_attacks_right:
                change = self.config.momentum_change_strength if team_intent < 0 else -self.config.momentum_change_strength
            else:
                change = self.config.momentum_change_strength if team_intent > 0 else -self.config.momentum_change_strength
            
            self.momentum = np.clip(self.momentum + change, -self.config.momentum_max, self.config.momentum_max)
        else:
            self.momentum *= self.config.momentum_decay
        
        # State flip logic
        if self.momentum > self.config.momentum_threshold:
            self.current_possession = "Westy"
        elif self.momentum < -self.config.momentum_threshold:
            self.current_possession = "Opponent"
    
    def flip_attack_direction(self) -> None:
        """Flip the attacking direction (half-court swap)"""
        self.westy_attacks_right = not self.westy_attacks_right
        logger.info(f"Attack direction flipped: {'RIGHT' if self.westy_attacks_right else 'LEFT'}")
    
    def get_momentum_percentage(self) -> float:
        """Get momentum as percentage (-100 to 100)"""
        return self.momentum


# ====================================================================
# MAIN GAME ENGINE
# ====================================================================

class VarsityAIEngine:
    """Main game analytics engine"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.scout = TimeoutScout(config)
        self.player_tracker = PlayerTracker(config)
        self.court_calibrator = CourtCalibrator(config)
        self.flow_tracker = OpticalFlowTracker(config)
        self.possession = PossessionEngine(config)
        
        self.model: Optional[YOLO] = None
        self.active_pnrs: Dict[int, Dict] = {}
        self.report_flash_timer = 0
    
    def _init_model(self) -> bool:
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.abspath(".")
            
            model_path = os.path.join(base_path, self.config.model_name)
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _crop_jersey_region(self, frame: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
        """Crop just the jersey region from a player bounding box"""
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        
        jersey_y1 = y1 + int(h * self.config.jersey_top_pct)
        jersey_y2 = y1 + int(h * self.config.jersey_bottom_pct)
        jersey_x1 = x1 + int(w * self.config.jersey_side_trim_pct)
        jersey_x2 = x2 - int(w * self.config.jersey_side_trim_pct)
        
        height, width = frame.shape[:2]
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(height, jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(width, jersey_x2)
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None
        
        return frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
    
    def _calculate_qsq(self, player_pos: Tuple[int, int], opponents: List[Dict]) -> float:
        """Calculate shot quality based on distance to closest defender"""
        if not opponents:
            return 50.0
        
        px, py = player_pos
        closest_dist = min(
            np.sqrt((px - o['x'])**2 + (py - o['y'])**2) 
            for o in opponents
        )
        
        # Normalize to 0-100 (assuming 250px = wide open)
        qsq = max(self.config.qsq_min, min(self.config.qsq_max, 
                  int((closest_dist / 250) * 100)))
        return qsq
    
    def _process_detections(self, frame: np.ndarray, frame_count: int,
                        t1_range: Tuple, t2_range: Tuple) -> Tuple[List[Dict], List[Dict], List[float]]:
        """Run YOLO detection and process all players with STABLE tracking"""
        westy_players = []
        opp_players = []
        pack_momentum = []
        
        if frame_count % self.config.detection_interval != 0:
            return westy_players, opp_players, pack_momentum
        
        try:
            results = self.model(frame, classes=[0], conf=0.4, verbose=False)
        except Exception as e:
            logger.warning(f"Detection failed: {e}")
            return westy_players, opp_players, pack_momentum
        
        if results[0].boxes is None:
            return westy_players, opp_players, pack_momentum
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        # Initialize tracker storage if not exists
        if not hasattr(self, 'tracked_players'):
            self.tracked_players = {}
            self.next_id = 0
            self.inactive_count = {}
        
        current_ids = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Find best match to existing player
            best_id = None
            best_score = -1
            
            for pid, pdata in self.tracked_players.items():
                if pdata['last_box'] is not None:
                    lx1, ly1, lx2, ly2 = pdata['last_box']
                    # Calculate IOU
                    ix1 = max(x1, lx1)
                    iy1 = max(y1, ly1)
                    ix2 = min(x2, lx2)
                    iy2 = min(y2, ly2)
                    if ix2 > ix1 and iy2 > iy1:
                        intersection = (ix2 - ix1) * (iy2 - iy1)
                        union = (x2-x1)*(y2-y1) + (lx2-lx1)*(ly2-ly1) - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        # Distance score
                        last_pos = pdata['positions'][-1] if pdata['positions'] else (cx, cy)
                        dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                        dist_score = 1.0 - min(1.0, dist / 100)
                        
                        score = iou * 0.6 + dist_score * 0.4
                        
                        if score > best_score and score > 0.3:
                            best_score = score
                            best_id = pid
            
            if best_id is not None:
                player_id = best_id
                self.tracked_players[player_id]['positions'].append((cx, cy))
                self.tracked_players[player_id]['last_box'] = (x1, y1, x2, y2)
                self.inactive_count[player_id] = 0
            else:
                player_id = self.next_id
                self.next_id += 1
                self.tracked_players[player_id] = {
                    'positions': deque(maxlen=30),
                    'team_votes': {'Westy': 0, 'Opponent': 0},
                    'team_locked': None,
                    'last_box': (x1, y1, x2, y2)
                }
                self.tracked_players[player_id]['positions'].append((cx, cy))
                self.inactive_count[player_id] = 0
            
            current_ids.append(player_id)
            
            # Color detection
            crop = self._crop_jersey_region(frame, [x1, y1, x2, y2])
            if crop is not None:
                team_guess = self.player_tracker.color_detector.detect_team(crop, t1_range, t2_range)
                if team_guess != "Unknown":
                    self.tracked_players[player_id]['team_votes'][team_guess] += 1
            
            # Determine locked team
            team = self.tracked_players[player_id]['team_locked']
            if team is None:
                westy_votes = self.tracked_players[player_id]['team_votes']['Westy']
                opp_votes = self.tracked_players[player_id]['team_votes']['Opponent']
                if westy_votes > 10:
                    team = "Westy"
                    self.tracked_players[player_id]['team_locked'] = team
                elif opp_votes > 10:
                    team = "Opponent"
                    self.tracked_players[player_id]['team_locked'] = team
                else:
                    team = "Unknown"
            
            # Calculate movement for momentum
            positions = self.tracked_players[player_id]['positions']
            if len(positions) > 1 and team != "Unknown":
                dx = cx - positions[-2][0]
                if abs(dx) > 2:
                    pack_momentum.append(dx)
            
            # Build player data
            player_data = {
                'id': player_id,
                'x': cx,
                'y': y2,
                'h': y2 - y1,
                'box': [x1, y1, x2, y2],
                'team': team
            }
            
            if team == "Westy":
                westy_players.append(player_data)
            elif team == "Opponent":
                opp_players.append(player_data)
        
        # Clean up old players
        for pid in list(self.tracked_players.keys()):
            if pid not in current_ids:
                self.inactive_count[pid] = self.inactive_count.get(pid, 0) + 1
                if self.inactive_count[pid] > 30:
                    del self.tracked_players[pid]
                    del self.inactive_count[pid]
        
        return westy_players, opp_players, pack_momentum
    
    def _draw_player_boxes(self, frame: np.ndarray, westy_players: List[Dict], 
                          opp_players: List[Dict]) -> None:
        """Draw player bounding boxes and qSQ labels"""
        # Calculate qSQ for all players
        for player in westy_players:
            player['qsq'] = self._calculate_qsq((player['x'], player['y']), opp_players)
            color = (0, 255, 255)  # Yellow for Westy
            label = f"W{player['id']}"
            qsq_color = (0, 255, 0) if player['qsq'] >= 30 else (0, 0, 255)
            
            cv2.rectangle(frame, (player['box'][0], player['box'][1]), 
                         (player['box'][2], player['box'][3]), color, 2)
            cv2.putText(frame, f"{label} qSQ:{player['qsq']:.0f}%", 
                       (player['box'][0], player['box'][1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, qsq_color, 2)
        
        for player in opp_players:
            player['qsq'] = self._calculate_qsq((player['x'], player['y']), westy_players)
            color = (0, 0, 255)  # Red for Opponent
            label = f"OPP{player['id']}"
            
            cv2.rectangle(frame, (player['box'][0], player['box'][1]), 
                         (player['box'][2], player['box'][3]), color, 2)
            cv2.putText(frame, f"{label}", 
                       (player['box'][0], player['box'][1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _detect_pick_and_roll(self, frame: np.ndarray, offensive_players: List[Dict]) -> None:
        """Detect pick and rolls"""
        # ADD THIS ONE LINE FOR DEBUG:
        cv2.putText(frame, f"PnR: {len(offensive_players)} players", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Create persistence tracker if it doesn't exist
        if not hasattr(self, '_pnr_tracker'):
            self._pnr_tracker = {}
        
        seen_pairs = set()
        PNR_DISTANCE = 100  # pixels
        NEEDED_FRAMES = 10  # about 0.5-1 second
        
        for i, p1 in enumerate(offensive_players):
            for p2 in offensive_players[i+1:]:
                dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                pair_key = f"{p1['id']}_{p2['id']}"
                
                if dist < PNR_DISTANCE:
                    seen_pairs.add(pair_key)
                    
                    if pair_key not in self._pnr_tracker:
                        self._pnr_tracker[pair_key] = 1
                    else:
                        self._pnr_tracker[pair_key] += 1
                    
                    # Only draw and trigger if they've been close long enough
                    if self._pnr_tracker[pair_key] >= NEEDED_FRAMES:
                        cv2.line(frame, (p1['x'], p1['y']), (p2['x'], p2['y']), (255, 0, 255), 3)
                        cv2.circle(frame, ((p1['x']+p2['x'])//2, (p1['y']+p2['y'])//2), 90, (255, 0, 255), 2)
                        
                        if self._pnr_tracker[pair_key] == NEEDED_FRAMES:
                            cv2.putText(frame, "PICK & ROLL!", (p1['x']-50, p1['y']-60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        # Start evaluation
                        handler = p1 if p1.get('qsq', 100) < p2.get('qsq', 100) else p2
                        if handler['id'] not in self.active_pnrs:
                            self.active_pnrs[handler['id']] = {
                                'combo': (p1['id'], p2['id']),
                                'start_qsq': handler.get('qsq', 50),
                                'timer': self.config.pnr_evaluation_frames,
                                'team': self.possession.current_possession
                            }
                else:
                    if pair_key in self._pnr_tracker:
                        self._pnr_tracker[pair_key] = max(0, self._pnr_tracker[pair_key] - 2)
                        if self._pnr_tracker[pair_key] == 0:
                            del self._pnr_tracker[pair_key]
        
        # Clean up
        for key in list(self._pnr_tracker.keys()):
            if key not in seen_pairs:
                self._pnr_tracker[key] = max(0, self._pnr_tracker[key] - 1)
                if self._pnr_tracker[key] == 0:
                    del self._pnr_tracker[key]
        
        # Existing evaluation logic (keep this part)
        completed = []
        for handler_id, pnr in self.active_pnrs.items():
            pnr['timer'] -= 1
            current_handler = next((p for p in offensive_players if p['id'] == handler_id), None)
            if current_handler:
                if pnr['timer'] > 0:
                    cv2.putText(frame, f"EVAL PnR... {pnr['timer']}", 
                               (current_handler['x']-50, current_handler['y']-60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    current_qsq = current_handler.get('qsq', 50)
                    success = current_qsq > (pnr['start_qsq'] + self.config.pnr_success_threshold)
                    if pnr['team'] == "Westy":
                        self.scout.log_pnr(pnr['start_qsq'], current_qsq)
                    status = "PnR SUCCESS!" if success else "PnR DEFENDED"
                    color = (0, 255, 0) if success else (0, 0, 255)
                    cv2.putText(frame, status, (current_handler['x']-60, current_handler['y']-80),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 4)
                    completed.append(handler_id)
            else:
                completed.append(handler_id)
        
        for pid in completed:
            del self.active_pnrs[pid]

    
    def _detect_mismatches(self, frame: np.ndarray, westy_players: List[Dict], 
                          opp_players: List[Dict], frame_count: int) -> None:
        """Detect favorable defensive mismatches"""
        if self.possession.current_possession != "Westy":
            return
        
        for westy in westy_players:
            for opp in opp_players:
                dist = np.sqrt((westy['x'] - opp['x'])**2 + (westy['y'] - opp['y'])**2)
                height_ratio = westy['h'] / max(1, opp['h'])
                vertical_sep = opp['box'][1] - westy['box'][1]
                
                mismatch_conditions = (
                    self.config.mismatch_min_dist < dist < self.config.mismatch_max_dist and
                    self.config.mismatch_height_ratio_min < height_ratio < self.config.mismatch_height_ratio_max and
                    vertical_sep > self.config.mismatch_vertical_sep
                )
                
                if mismatch_conditions:
                    cv2.line(frame, (westy['x'], westy['y']), (opp['x'], opp['y']), (0, 165, 255), 3)
                    cv2.putText(frame, "MISMATCH!", (westy['x']-40, westy['box'][1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    if frame_count % 15 == 0:
                        self.scout.log_mismatch()
    
    def _detect_cramping(self, frame: np.ndarray, westy_players: List[Dict], 
                        width: int, height: int, frame_count: int) -> None:
        """Detect offensive spacing issues (cramping)"""
        if self.possession.current_possession != "Westy" or len(westy_players) < 3:
            return
        
        westy_sorted = sorted(westy_players, key=lambda w: w['x'])
        off_width = westy_sorted[-1]['x'] - westy_sorted[0]['x']
        lowest_y = max(w['y'] for w in westy_players) + 20
        
        # Draw spread line
        cv2.line(frame, (westy_sorted[0]['x'], lowest_y), 
                (westy_sorted[-1]['x'], lowest_y), (255, 255, 255), 2)
        cv2.putText(frame, f"OFFENSIVE SPREAD: {off_width}px", 
                   (westy_sorted[0]['x'] + 10, lowest_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if frame_count % 30 == 0:
            self.scout.log_spacing(off_width)
        
        # Check for cramping
        if off_width < self.config.cramped_spread_threshold:
            avg_x = np.mean([w['x'] for w in westy_players])
            
            if self.possession.westy_attacks_right:
                is_deep_rim = avg_x > width * (1 - self.config.rim_region_pct)
                is_middle = (width * self.config.middle_region_pct[0] < avg_x <= 
                            width * self.config.middle_region_pct[1])
            else:
                is_deep_rim = avg_x < width * self.config.rim_region_pct
                is_middle = (width * self.config.middle_region_pct[0] <= avg_x < 
                            width * self.config.middle_region_pct[1])
            
            if is_deep_rim:
                avg_height = np.mean([w['h'] for w in westy_players])
                rim_region_pct = 0.70 if self.possession.westy_attacks_right else 0.30
                if self.possession.westy_attacks_right:
                    deep_players = [w for w in westy_players if w['x'] > width * rim_region_pct]
                else:
                    deep_players = [w for w in westy_players if w['x'] < width * rim_region_pct]
                
                short_and_deep = any(p['h'] < avg_height * 0.9 for p in deep_players)
                cramp_type = "RIM_CRAMP_GUARDS" if short_and_deep else "RIM_CRAMP"
                
                cv2.putText(frame, f"CRAMPED: {'GUARDS INSIDE!' if short_and_deep else 'UNDER RIM!'}",
                           (westy_sorted[-1]['x'] + 10, lowest_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if frame_count % 30 == 0:
                    self.scout.log_cramp(cramp_type)
            
            elif is_middle:
                cv2.putText(frame, "STAGNANT: DRIVE MORE!", 
                           (westy_sorted[-1]['x'] + 10, lowest_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                if frame_count % 30 == 0:
                    self.scout.log_cramp("MIDDLE_STAGNANT")
    
    def _detect_defensive_shell(self, frame: np.ndarray, defending_team: List[Dict], 
                               current_possession: str) -> None:
        """Detect and classify defensive formation"""
        if len(defending_team) < 3:
            return
        
        def_color = (0, 0, 255) if current_possession == "Westy" else (0, 255, 255)
        def_name = "OPPONENT" if current_possession == "Westy" else "WESTY"
        
        def_pts = np.array([[p['x'], p['y']] for p in defending_team], np.int32)
        hull = cv2.convexHull(def_pts)
        cv2.polylines(frame, [hull], isClosed=True, color=def_color, thickness=2)
        
        frame_area = self.config.frame_width * self.config.frame_height
        is_compact = (cv2.contourArea(hull) / frame_area) < self.config.compact_shell_area_ratio
        shell_status = "COMPACT (Zone/Pack-Line)" if is_compact else "SPREAD (Man-to-Man)"
        
        cv2.putText(frame, f"{def_name} SHELL: {shell_status}", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, def_color, 2)
    
    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw heads-up display"""
        cv2.rectangle(frame, (10, 10), (450, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"OFFENSE: {self.possession.current_possession}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"WESTY ATTACKING: {'RIGHT' if self.possession.westy_attacks_right else 'LEFT'}",
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, "H: FLIP HALF | R: REPORT | Q: QUIT", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"MOMENTUM: {int(self.possession.get_momentum_percentage())}%", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self, t1_name: str, t1_range: Tuple, t2_name: str, t2_range: Tuple) -> None:
        print(f"Attempting to open camera with ID: {self.config.camera_id}")
        """Main game loop"""
        if not self._init_model():
            logger.error("Failed to initialize model, exiting")
            return
        
        # Ensure court is calibrated
        if self.court_calibrator.court_polygon is None:
            self.court_calibrator.calibrate()
        
        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        # Camera warmup
        for _ in range(self.config.camera_warmup_frames):
            cap.read()
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
            frame = normalize_lighting(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            y_cutoff = int(height * 0.22)
            
            # Update optical flow for camera compensation
            cam_dx, cam_dy = self.flow_tracker.update(gray, frame_count, y_cutoff)
            
            # Process detections (skip frames for performance)
            westy_players, opp_players, pack_momentum = self._process_detections(
                frame, frame_count, t1_range, t2_range
            )
            
            # Apply camera compensation to momentum
            adjusted_momentum = []
            for m in pack_momentum:
                adjusted_momentum.append(m - cam_dx)
            
            # Update possession
            self.possession.update(adjusted_momentum, self.possession.westy_attacks_right)
            
            # Log shot quality for offensive team
            if self.possession.current_possession == "Westy" and westy_players and frame_count % 15 == 0:
                avg_qsq = np.mean([p.get('qsq', 50) for p in westy_players])
                self.scout.log_shot_quality(avg_qsq)
            
            # Draw player boxes and labels
            self._draw_player_boxes(frame, westy_players, opp_players)
            
            # Advanced analytics
            offensive_players = westy_players if self.possession.current_possession == "Westy" else opp_players
            self._detect_pick_and_roll(frame, offensive_players)
            self._detect_mismatches(frame, westy_players, opp_players, frame_count)
            self._detect_cramping(frame, westy_players, width, height, frame_count)
            
            # Defensive shell detection
            defending_team = opp_players if self.possession.current_possession == "Westy" else westy_players
            self._detect_defensive_shell(frame, defending_team, self.possession.current_possession)
            
            # Draw HUD
            self._draw_hud(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.possession.flip_attack_direction()
            elif key == ord('r'):
                self.scout.generate_timeout_report()
                self.report_flash_timer = 40
            
            # Flash report confirmation
            if self.report_flash_timer > 0:
                cv2.putText(frame, "REPORT GENERATED!", (width//2 - 200, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
                self.report_flash_timer -= 1
            
            cv2.imshow('Varsity AI - Refined', frame)
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Game engine stopped")


# ====================================================================
# ENTRY POINT
# ====================================================================

def main():
    """Example usage"""
    config = AppConfig()
    engine = VarsityAIEngine(config)
    
    # Example color ranges (replace with actual values)
    westy_range = ([0, 100, 100], [10, 255, 255])  # Example: red
    opp_range = ([100, 100, 100], [130, 255, 255])  # Example: blue
    
    engine.run("Westy", westy_range, "Opponent", opp_range)

def get_camera():
    """Test which camera works"""
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return 0

if __name__ == "__main__":
    main()