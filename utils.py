class BallSettings:
    def __init__(self, ball_path=None, ball_low_hsv=None, ball_high_hsv=None):
        self.ball_path = ball_path
        self.ball_low_hsv = ball_low_hsv
        self.ball_high_hsv = ball_high_hsv
        self.hue_tolerance = 10
        self.sv_tolerance = 20
        self.is_set = False

GameBall = BallSettings()