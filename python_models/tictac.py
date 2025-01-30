import numpy as np
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == letter for s in row]):
            return True
        
        col_ind = square % 3
        col = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == letter for s in col]):
            return True
        
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([s == letter for s in diagonal2]):
                return True
        return False

class QLearningAgent:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q_table = defaultdict(dict)
        self.alpha = alpha
        self.epsilon = epsilon
        self.history = []

    def get_state(self, board):
        return tuple(board)

    def choose_action(self, state, available_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = [self.q_table[state].get(a, 0) for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)

    def update_q_table(self, reward):
        for (state, action) in self.history:
            current_q = self.q_table[state].get(action, 0)
            new_q = current_q + self.alpha * (reward - current_q)
            self.q_table[state][action] = new_q
        self.history = []

    def reset_history(self):
        self.history = []

def train(agent, episodes=10000):
    for episode in range(episodes):
        game = TicTacToe()
        agent.reset_history()
        current_player = 'X'
        
        while game.empty_squares():
            if current_player == 'X':
                state = agent.get_state(game.board)
                available_actions = game.available_moves()
                if not available_actions:
                    break
                action = agent.choose_action(state, available_actions)
                game.make_move(action, 'X')
                agent.history.append((state, action))
                
                if game.current_winner:
                    reward = 1
                    break
                current_player = 'O'
            else:
                available_actions = game.available_moves()
                if available_actions:
                    action = np.random.choice(available_actions)
                    game.make_move(action, 'O')
                    if game.current_winner:
                        reward = -1
                        break
                current_player = 'X'
        else:
            reward = 0
        
        agent.update_q_table(reward)
        agent.epsilon = max(0.01, agent.epsilon * 0.999)
    
    print("Training completed!")
    print(f"Learned {len(agent.q_table)} states")

def play_human(agent):
    game = TicTacToe()
    print("Welcome to Tic Tac Toe!")
    print("Board positions:")
    game.print_board_nums()
    print("\nYou're O, the AI is X. The AI goes first.")
    
    current_player = 'X'
    
    while game.empty_squares() and not game.current_winner:
        if current_player == 'X':
            state = agent.get_state(game.board)
            available_actions = game.available_moves()
            action = agent.choose_action(state, available_actions)
            game.make_move(action, 'X')
            print("\nAI's move:")
            game.print_board()
            current_player = 'O'
        else:
            available_actions = game.available_moves()
            while True:
                try:
                    action = int(input("\nYour move (0-8): "))
                    if action in available_actions:
                        break
                    print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 0-8")
            
            game.make_move(action, 'O')
            print("\nYour move:")
            game.print_board()
            current_player = 'X'
    
    if game.current_winner:
        if game.current_winner == 'X':
            print("\nAI wins!")
        else:
            print("\nYou win!")
    else:
        print("\nIt's a tie!")

if __name__ == "__main__":
    agent = QLearningAgent(alpha=0.5, epsilon=0.1)
    train(agent, episodes=10000)
    play_human(agent)