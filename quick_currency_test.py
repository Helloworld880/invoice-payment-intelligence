
import sys
import os
sys.path.append('.')
from src.utils.helpers import format_currency

print("Testing currency formatting:")
print(f"1000 -> '{format_currency(1000)}'")
print(f"500.50 -> '{format_currency(500.50)}'")
print(f"15000 -> '{format_currency(15000)}'")
print(f"2500000 -> '{format_currency(2500000)}'")