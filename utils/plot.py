import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    path = "expdir/" + "adult" + "/"
    path = path + "train_start_vae.txt"
    RMSE_list = []
    epoch_list = []
    Loss_list = []
    MSE_list = []
    BCE_list = []
    EnLoss_list = [] #KL
    DeLoss_list = []
    G_loss = []
    L_loss = []
    KL_list = []
    All_loss_list = []
    D_list = []
    G_D_list = []
    with open(path, "r") as f:
        flag = -1
        for line in f:
            str = "RMSE：  "
            RMSE_pos = line.find(str)
            if RMSE_pos != -1:
                data = float(line[RMSE_pos + len(str):].strip()[:6])
                RMSE_list.append(data)# 获取冒号后面的字符串（去除前导和末尾空格）
            str = "MSE：  "
            Loss_pos = line.find(str)
            if Loss_pos != -1:
                data = float(line[Loss_pos + len(str):].strip()[:6])
                MSE_list.append(data)

            str = "BCE：  "
            Loss_pos = line.find(str)
            if Loss_pos != -1:
                data = float(line[Loss_pos + len(str):].strip()[:6])
                BCE_list.append(data)

            str = "KL：  "
            Loss_pos = line.find(str)
            if Loss_pos != -1:
                data = float(line[Loss_pos + len(str):].strip()[:6])
                KL_list.append(data)

            str = "G_D_loss：  "
            G_pos = line.find(str)
            if G_pos != -1:
                data = float(line[G_pos + len(str):].strip()[:6])
                G_D_list.append(data)

            str = "All_loss:     "
            L_pos = line.find(str)
            if L_pos != -1:
                data = float(line[L_pos + len(str):].strip()[:6])
                All_loss_list.append(data)

            str = "D_loss:     "
            L_pos = line.find(str)
            if L_pos != -1:
                data = float(line[L_pos + len(str):].strip()[:6])
                D_list.append(data)

        DeLoss_list = list(np.array(MSE_list) + np.array(BCE_list))

        epoch_list = list(np.arange(len(RMSE_list)))
        plt.figure(1)
        plt.plot(epoch_list,RMSE_list)
        plt.title("RMSE")
        # add a label to the x-axis
        plt.xlabel("Epoch")
        # add a label to the y-axis
        plt.ylabel("RMSE")
        # show the plot
        plt.show()



        plt.figure(3)
        plt.plot(epoch_list,KL_list)
        plt.title("KL_Loss")
        # add a label to the x-axis
        plt.xlabel("Epoch")
        # add a label to the y-axis
        plt.ylabel("KL_Loss(Encoder)")
        # show the plot
        plt.show()

        plt.figure(4)
        plt.plot(epoch_list,BCE_list)
        plt.plot(epoch_list,MSE_list,'r')
        plt.plot(epoch_list,DeLoss_list,'b')
        plt.title("RLoss")
        # add a label to the x-axis
        plt.xlabel("Epoch")
        # add a label to the y-axis
        plt.ylabel("R(Decoder)")
        # show the plot
        plt.show()

        plt.figure(5)
        plt.plot(epoch_list,G_D_list)
        plt.title("G_D_Loss")
        # add a label to the x-axis
        plt.xlabel("Epoch")
        # add a label to the y-axis
        plt.ylabel("G_D__loss")
        # show the plot
        plt.show()

        plt.figure(2)
        plt.plot(epoch_list,All_loss_list)
        plt.title("All_G_Loss")
        # add a label to the x-axis
        plt.xlabel("Epoch")
        # add a label to the y-axis
        plt.ylabel("Loss")
        # show the plot
        plt.show()

        plt.figure(6)
        plt.plot(epoch_list,D_list)
        plt.title("D_Loss")
        # add a label to the x-axis
        plt.xlabel("Epoch")
        # add a label to the y-axis
        plt.ylabel("D_loss")
        # show the plot
        plt.show()


