import os, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphGenerator(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(6, 4))

    def plot_weather_temp_train(self, train_x, train_y, val_x, val_y, pred_val_y):

        #fig = plt.figure(figsize=(6, 4))
        ax = self.fig.add_subplot(1, 1, 1)

        val_label_scatter = ax.scatter(val_x, val_y, color='black')
        val_pred_scatter = ax.scatter(val_x, pred_val_y, color='red')
        train_label_scatter = ax.scatter(train_x, train_y, marker='o', color='blue')

        plt.xlabel("month")
        plt.ylabel("temperature")

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)

        plt.ylim([0, 35])

        #plt.savefig("results/" + str(d_now.strftime('%Y%m%d_%H%M%S')) + ".png")
        #plt.close(fig)
    
        return val_label_scatter, val_pred_scatter, train_label_scatter

        
    def plot_weather_temp_test(self, train_x, train_y, test_x, test_y, pred_test_y, d_now):

        save_path = str(d_now.strftime('%Y%m%d_%H%M%S'))
            
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
            
        ax.scatter(test_x, test_y, color='black')
        ax.scatter(test_x, pred_test_y, color='red')
        ax.scatter(train_x, train_y, marker='o', color='blue')
            
        plt.xlabel("month")
        plt.ylabel("temperature")
            
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
            
        plt.ylim([0, 35])

        if not os.path.exists("results/" + str(d_now.strftime('%Y%m%d_%H%M%S'))):
            os.mkdir("results/" + str(d_now.strftime('%Y%m%d_%H%M%S')))
            
        plt.savefig("results/" + str(d_now.strftime('%Y%m%d_%H%M%S')) + "/" + str(d_now.strftime('%Y%m%d_%H%M%S')) + ".png")
        plt.close(fig)
            
        return
            
    def gif_generator(self, imgs, d_now):
        if not os.path.exists("results/" + str(d_now.strftime('%Y%m%d_%H%M%S'))):
            os.mkdir("results/" + str(d_now.strftime('%Y%m%d_%H%M%S')))

        save_path = "results/" + str(d_now.strftime('%Y%m%d_%H%M%S')) + "/" + str(d_now.strftime('%Y%m%d_%H%M%S')) + ".gif"
        ani = animation.ArtistAnimation(self.fig, imgs, interval=100, blit=True, repeat_delay=1000)
        ani.save(save_path, writer="imagemagick")
        plt.show()
        plt.close(self.fig)
    
    def loss_graph(self, losses, d_now):
        if not os.path.exists("results/" + str(d_now.strftime('%Y%m%d_%H%M%S'))):
            os.mkdir("results/" + str(d_now.strftime('%Y%m%d_%H%M%S')))
        
        fig = plt.figure()
        
        plt.plot(losses)
        plt.xlabel('epoch')

        plt.savefig("results/" + str(d_now.strftime('%Y%m%d_%H%M%S')) + "/losses_" + str(d_now.strftime('%Y%m%d_%H%M%S')) + ".png")
        
        plt.legend()
        plt.close(fig)
        