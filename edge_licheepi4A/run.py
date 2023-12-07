import os

try:
    for i in range(21):
        os.system(f"python testmobile.py --model alexnet --cut {i} --order back")

    for i in range(47):
        os.system(f"python testmobile.py --model vgg16 --cut {i} --order back")
except KeyboardInterrupt:
    print("KeyboardInterrupt")

