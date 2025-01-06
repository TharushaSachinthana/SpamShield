import os

if __name__ == "__main__":
    os.system("python preprocess.py")
    os.system("python train.py")
    os.system("python predict.py")
    print("Pipeline completed.")
