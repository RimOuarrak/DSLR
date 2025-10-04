from logreg_class import LogisticRegression
import csv
import sys
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)

if len(sys.argv) != 2:
		print("Usage: python logreg_train.py dataset_train.csv")
		sys.exit(1)

dataset_file = sys.argv[1]
skip_cols = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]

X = []
y = []

with open(dataset_file, 'r') as f:
	reader = csv.DictReader(f)
	feature_names = [col for col in reader.fieldnames if col not in skip_cols]

	for row in reader:
		try:
			features = [float(row[f]) for f in feature_names]  # all must convert
			X.append(features)
			y.append(row["Hogwarts House"])
		except (ValueError, TypeError):
			# Skip this row entirely if any feature is non-numeric
			continue

def select_algorithm():
	"""Interactive arrow-based algorithm selection"""
	import termios
	import tty
	
	options = [
		("Full-Batch Gradient Descent (BGD)", "Uses entire dataset for each update", Fore.CYAN),
		("Stochastic Gradient Descent (SGD)", "Uses one sample at a time", Fore.MAGENTA),
		("Mini-Batch Gradient Descent (MBGD)", "Uses small batches of data", Fore.YELLOW)
	]
	
	selected = 0
	
	def get_key():
		"""Get a single keypress from stdin"""
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(fd)
			key = sys.stdin.read(1)
			if key == '\x1b':  # ESC sequence
				key += sys.stdin.read(2)
			return key
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	
	def display_menu():
		"""Display the selection menu"""
		print(f"\n{Fore.BLUE}🚀 Select Gradient Descent Algorithm:{Style.RESET_ALL}")
		print(f"{Fore.WHITE}Use ↑/↓ arrows to navigate, Enter to select, Ctrl+C to cancel{Style.RESET_ALL}\n")
		
		for i, (name, desc, color) in enumerate(options):
			if i == selected:
				print(f"{color}▶ {name}{Style.RESET_ALL} - {desc}")
			else:
				print(f"  {color}{name}{Style.RESET_ALL} - {desc}")
	
	try:
		while True:
			# Clear screen and display menu
			print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
			display_menu()
			
			key = get_key()
			
			if key == '\x1b[A':  # Up arrow
				selected = (selected - 1) % len(options)
			elif key == '\x1b[B':  # Down arrow
				selected = (selected + 1) % len(options)
			elif key == '\r' or key == '\n':  # Enter
				break
			elif key == '\x03':  # Ctrl+C
				raise KeyboardInterrupt
		
		# Clear screen and show selection
		print("\033[2J\033[H", end="")
		selected_option = options[selected]
		print(f"{selected_option[2]}✅ Selected: {selected_option[0]}{Style.RESET_ALL}\n")
		
		if selected == 0:  # Full-Batch GD
			return LogisticRegression(learning_rate=0.1, epochs=1000)
		elif selected == 1:  # SGD
			return LogisticRegression(learning_rate=0.1, epochs=1000, batch_size=1)
		else:  # Mini-Batch GD
			batch_size = get_batch_size()
			return LogisticRegression(learning_rate=0.1, epochs=1000, batch_size=batch_size)
			
	except KeyboardInterrupt:
		print(f"\n{Fore.RED}Training cancelled by user.{Style.RESET_ALL}")
		sys.exit(0)
	except Exception:
		# Fallback to numbered selection if arrow keys don't work
		print(f"{Fore.YELLOW}⚠️  Arrow keys not supported, using numbered selection...{Style.RESET_ALL}\n")
		return select_algorithm_fallback()

def select_algorithm_fallback():
	"""Fallback numbered selection when arrow keys don't work"""
	print(f"\n{Fore.BLUE}🚀 Select Gradient Descent Algorithm:{Style.RESET_ALL}")
	print(f"{Fore.CYAN}1. Full-Batch Gradient Descent (BGD){Style.RESET_ALL} - Uses entire dataset for each update")
	print(f"{Fore.MAGENTA}2. Stochastic Gradient Descent (SGD){Style.RESET_ALL} - Uses one sample at a time")
	print(f"{Fore.YELLOW}3. Mini-Batch Gradient Descent (MBGD){Style.RESET_ALL} - Uses small batches of data")
	
	while True:
		try:
			choice = input(f"\n{Fore.GREEN}Enter your choice (1-3): {Style.RESET_ALL}")
			
			if choice == '1':
				print(f"{Fore.CYAN}✅ Selected: Full-Batch Gradient Descent{Style.RESET_ALL}")
				return LogisticRegression(learning_rate=0.1, epochs=1000)
			elif choice == '2':
				print(f"{Fore.MAGENTA}✅ Selected: Stochastic Gradient Descent{Style.RESET_ALL}")
				return LogisticRegression(learning_rate=0.1, epochs=1000, batch_size=1)
			elif choice == '3':
				batch_size = get_batch_size()
				print(f"{Fore.YELLOW}✅ Selected: Mini-Batch Gradient Descent (batch_size={batch_size}){Style.RESET_ALL}")
				return LogisticRegression(learning_rate=0.1, epochs=1000, batch_size=batch_size)
			else:
				print(f"{Fore.RED}❌ Invalid choice! Please enter 1, 2, or 3.{Style.RESET_ALL}")
		except KeyboardInterrupt:
			print(f"\n{Fore.RED}Training cancelled by user.{Style.RESET_ALL}")
			sys.exit(0)

def get_batch_size():
	"""Get batch size for mini-batch gradient descent"""
	while True:
		try:
			batch_size = input(f"{Fore.YELLOW}Enter batch size (default 32): {Style.RESET_ALL}")
			if batch_size == '':
				return 32
			batch_size = int(batch_size)
			if batch_size > 0:
				return batch_size
			else:
				print(f"{Fore.RED}❌ Batch size must be positive!{Style.RESET_ALL}")
		except ValueError:
			print(f"{Fore.RED}❌ Please enter a valid number!{Style.RESET_ALL}")
		except KeyboardInterrupt:
			print(f"\n{Fore.RED}Training cancelled by user.{Style.RESET_ALL}")
			sys.exit(0)

# Select algorithm interactively
model = select_algorithm()

model.fit(X, y)
model.save_weights("weights.csv")
