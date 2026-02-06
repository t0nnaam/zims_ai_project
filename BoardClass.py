

class Board: 
    def __init__(self, n, terminal_states, initial_state, obstacles):
        self.board = [[0 for _ in len(n)] for _ in len(n)]
        self.terminal_states = terminal_states
        self.root = initial_state
        self.visited = []
        self.obstacles = obstacles
        self.utilities = [[0 for _ in len(n)] for _ in len(n)]
        self.n = n

    @property
    def obstacles():
        return self.obstacles

    def is_valid_coordinate((x_val, y_val)):
        if self.n <= x_val  or x_val < 0:
            return False
        elif self.n <= y_val  or y_val < 0:
            return False

        return True
        
    def compute_coordinates((x_val, y_val)):
            x1, x2 = x_val + 1, x_val - 1
            y1, y2 = y_val + 1, y_val - 1
            coordinates = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    

    @obstacles.setter
    def set_obstacles(self, obstacle_path):
        self.obstacles = obstacle_path


    def utility():
        #minus the states near the obstacles
        for x0, y0 in obstacles:
            coordinates = compute_coordinates((x0, y0))
            for xn, yn in coordinates:
                if is_valid_coordinate((xn, yn)):
                    self.utilities[xn][yn] -= 5

        # plus the states that get us to the terminal state 
        for xs, ys in terminal_states:
            coordinates = compute_coordinates((xs, ys))
            for xn, yn in coordinates:
                if is_valid_coordinate((xn, yn)):
                    self.utilities[xn][yn] += 10



