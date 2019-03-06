import crnninterface
import eastinterface

def main():
    mycrnn = crnninterface.crnnclass()
    myeast = eastinterface.eastclass()
    myeast.east_detect()
    mycrnn.crnn_detect()

if __name__ == '__main__':
    main()