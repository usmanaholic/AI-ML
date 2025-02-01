import tkinter as tk
from tkinter import messagebox
import numpy as np
from collections import defaultdict
import copy

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_winner = self.current_winner
        return new_game

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

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

def get_symmetries(board):
    symmetries = []
    symmetries.append(board)
    symmetries.append([board[6], board[3], board[0], board[7], board[4], board[1], board[8], board[5], board[2]])
    symmetries.append([board[8], board[7], board[6], board[5], board[4], board[3], board[2], board[1], board[0]])
    symmetries.append([board[2], board[5], board[8], board[1], board[4], board[7], board[0], board[3], board[6]])
    symmetries.append([board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]])
    symmetries.append([board[6], board[7], board[8], board[3], board[4], board[5], board[0], board[1], board[2]])
    symmetries.append([board[0], board[3], board[6], board[1], board[4], board[7], board[2], board[5], board[8]])
    symmetries.append([board[8], board[5], board[2], board[7], board[4], board[1], board[6], board[3], board[0]])
    return symmetries

class QLearningAgent:
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=0.9):
        self.q_table = defaultdict(dict)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.history = []

    def get_state(self, board):
        symmetries = get_symmetries(board)
        min_state = min([tuple(s) for s in symmetries])
        return min_state

    def choose_action(self, state, available_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = [self.q_table[state].get(a, 0) for a in available_actions]
            max_q = max(q_values) if q_values else 0
            best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return np.random.choice(best_actions) if best_actions else np.random.choice(available_actions)

    def update_q_table(self, reward):
        self.history.reverse()
        next_max = 0
        for (state, action) in self.history:
            current_q = self.q_table[state].get(action, 0)
            new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
            self.q_table[state][action] = new_q
            next_max = max(self.q_table[state].values()) if self.q_table[state] else 0
        self.history = []

class TicTacToeGUI:
    def __init__(self, agent):
        self.agent = agent
        self.game = TicTacToe()
        self.human_symbol = ''
        self.ai_symbol = ''
        
        self.setup_window = tk.Tk()
        self.setup_window.title("Choose Symbol")
        tk.Label(self.setup_window, text="Choose your symbol:", font=('Arial', 14)).pack(pady=10)
        tk.Button(self.setup_window, text='X', font=('Arial', 20), width=4, 
                 command=lambda: self.start_game('X')).pack(side=tk.LEFT, padx=20)
        tk.Button(self.setup_window, text='O', font=('Arial', 20), width=4,
                 command=lambda: self.start_game('O')).pack(side=tk.RIGHT, padx=20)
        self.setup_window.mainloop()

    def start_game(self, symbol):
        self.setup_window.destroy()
        self.human_symbol = symbol
        self.ai_symbol = 'O' if symbol == 'X' else 'X'
        
        self.window = tk.Tk()
        self.window.title(f"Tic Tac Toe - You: {self.human_symbol} vs AI: {self.ai_symbol}")
        
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.window, text='', font=('Arial', 40), width=4, height=2,
                          command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
        
        self.status_label = tk.Label(self.window, text="", font=('Arial', 14))
        self.status_label.grid(row=3, columnspan=3)
        
        if self.human_symbol == 'X':
            self.status_label.config(text="Your turn (X)")
            for btn in self.buttons:
                btn.config(state='normal')
        else:
            self.status_label.config(text="AI's turn")
            self.ai_turn()
        
        self.window.mainloop()

    def ai_turn(self):
        self.status_label.config(text="AI is thinking...")
        self.window.update_idletasks()
        self.window.update()
        
        available_actions = self.game.available_moves()
        if not available_actions:
            self.game_over("It's a tie!")
            return
        
        # Check for immediate win
        for action in available_actions:
            game_copy = self.game.copy()
            if game_copy.make_move(action, self.ai_symbol):
                if game_copy.current_winner:
                    self.game.make_move(action, self.ai_symbol)
                    self.buttons[action].config(text=self.ai_symbol, 
                                              fg='red' if self.ai_symbol == 'X' else 'green',
                                              state='disabled')
                    self.check_game_end()
                    return
        
        # Check for opponent's immediate win
        for action in available_actions:
            game_copy = self.game.copy()
            if game_copy.make_move(action, self.human_symbol):
                if game_copy.current_winner:
                    self.game.make_move(action, self.ai_symbol)
                    self.buttons[action].config(text=self.ai_symbol, 
                                              fg='red' if self.ai_symbol == 'X' else 'green',
                                              state='disabled')
                    self.check_game_end()
                    return
        
        # Proceed with Q-Learning
        current_board = self.game.board.copy()
        if self.ai_symbol == 'O':
            mirrored_board = ['X' if c == 'O' else 'O' if c == 'X' else ' ' for c in current_board]
            state = self.agent.get_state(mirrored_board)
        else:
            state = self.agent.get_state(current_board)
        
        action = self.agent.choose_action(state, available_actions)
        self.game.make_move(action, self.ai_symbol)
        self.buttons[action].config(text=self.ai_symbol, 
                                  fg='red' if self.ai_symbol == 'X' else 'green',
                                  state='disabled')
        self.check_game_end()

    def check_game_end(self):
        if self.game.current_winner:
            self.game_over("AI wins!" if self.game.current_winner == self.ai_symbol else "You win!")
        elif not self.game.available_moves():
            self.game_over("It's a tie!")
        else:
            self.status_label.config(text=f"Your turn ({self.human_symbol})")
            for btn in self.buttons:
                btn.config(state='normal' if btn['text'] == '' else 'disabled')

    def human_move(self, square):
        if self.game.make_move(square, self.human_symbol):
            self.buttons[square].config(text=self.human_symbol, 
                                      fg='blue', state='disabled')
            self.check_game_end()
            if not self.game.current_winner and self.game.available_moves():
                for btn in self.buttons:
                    btn.config(state='disabled')
                self.window.after(500, self.ai_turn)

    def game_over(self, message):
        for btn in self.buttons:
            btn.config(state='disabled')
        answer = messagebox.askyesno("Game Over", f"{message}\nPlay again?")
        if answer:
            self.reset_game()
        else:
            self.window.destroy()

    def reset_game(self):
        self.game = TicTacToe()
        for btn in self.buttons:
            btn.config(text='', state='normal')
        
        if self.human_symbol == 'X':
            self.status_label.config(text="Your turn (X)")
            for btn in self.buttons:
                btn.config(state='normal')
        else:
            for btn in self.buttons:
                btn.config(state='disabled')
            self.status_label.config(text="AI's turn")
            self.ai_turn()

def train(agent, episodes=10000):
    for _ in range(episodes):
        game = TicTacToe()
        agent.history = []
        current_player = 'X'
        reward = 0
        
        while True:
            state = agent.get_state(game.board)
            available_actions = game.available_moves()
            if not available_actions:
                break
            
            if current_player == 'X':
                action = agent.choose_action(state, available_actions)
                game.make_move(action, 'X')
                agent.history.append((state, action))
                
                if game.current_winner:
                    reward = 1
                    break
                current_player = 'O'
            else:
                # Smart opponent: win if possible, else block, else random
                winning_actions = []
                for a in available_actions:
                    game_copy = game.copy()
                    game_copy.make_move(a, 'O')
                    if game_copy.current_winner:
                        winning_actions.append(a)
                if winning_actions:
                    action = np.random.choice(winning_actions)
                else:
                    blocking_actions = []
                    for a in available_actions:
                        game_copy = game.copy()
                        game_copy.make_move(a, 'X')
                        if game_copy.current_winner:
                            blocking_actions.append(a)
                    if blocking_actions:
                        action = np.random.choice(blocking_actions)
                    else:
                        action = np.random.choice(available_actions)
                
                game.make_move(action, 'O')
                if game.current_winner:
                    reward = -1
                    break
                current_player = 'X'
        
        agent.update_q_table(reward)
        agent.epsilon = max(0.01, agent.epsilon * 0.999)

if __name__ == "__main__":
    agent = QLearningAgent(epsilon=0.1)
    train(agent, episodes=20000)
    agent.epsilon = 0
    TicTacToeGUI(agent)