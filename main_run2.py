
import os
for i in range(15, 32):
	print("the target index is", i)
	over = os.system("python Infor_mine3.py --target_index=" + str(i))