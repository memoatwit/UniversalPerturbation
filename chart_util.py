import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from itertools import product
import os
import re

class Chart_Util:

    def __init__(self, image_dir = "ch_images/", orig_label_name = "lemon", epsilons = [0.50, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01, 0.005, 0.0]):
        self.image_dir = image_dir
        self.orig_label_name = orig_label_name
        self.epsilons = epsilons

    def plot(self):

        attacks_name = ["fgsm" , "iterative", "universal"]

        original_images = ['ch_images_angle_0.jpg', 'ch_images_angle_1.jpg', 'ch_images_angle_2.jpg', 'ch_images_angle_3.jpg', 'ch_images_angle_4.jpg']

        self.epsilons.reverse()

        log_data = pd.read_csv(self.image_dir + 'results_' + self.orig_label_name + '.csv')


        # Number of rows and columns for the subplot grid
        num_rows = 1 + len(original_images)
        num_columns = len(self.epsilons)  # 1 for the original images, rest for adversarial images

        # Create a figure with subplots
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3, num_rows * 3))


        for attack in attacks_name:
            attack_log = log_data[log_data['name'].str.contains(self.orig_label_name + "_" + attack)]
            print(attack_log)

            for col, eps in enumerate(self.epsilons, start=0):
                axs[0, col].text(0.35, -0.15, 'Epsilon ' + str(eps))
                axs[0, col].axis('off')  # Hide the axis

            for row, orig_image_name in enumerate(original_images, start=1):
                # Load and display the original image in the first column
                orig_image_path = os.path.join(self.image_dir, orig_image_name)
                orig_image = mpimg.imread(orig_image_path)
                match = re.search(r'angle_(\d+)', orig_image_name)
                im_no = int(match.group(1))

                #axs[row, 0].imshow(orig_image)
                #axs[row, 0].set_title('Original Image ' + str(im_no))
                #axs[row, 0].axis('off')  # Hide the axis

                match = re.search(r'angle_(\d+)', orig_image_name)
                im_no = int(match.group(1))
                # Load and display each adversarial image in the subsequent columns
                for col, eps in enumerate(self.epsilons, start=0):
                    adv_image_name = f"adv_{self.orig_label_name}_{attack}_img{im_no}_eps{int(eps*1000):04d}.jpg"
                    adv_image_path = os.path.join(self.image_dir, adv_image_name)
                    adv_image = mpimg.imread(adv_image_path)
                    axs[row, col].imshow(adv_image)

                    name = (adv_image_name.split('.')[0]).split('_')
                    log_name = name[1] + "_" + name[2] + "_train_" + name[3] + "_" + name[4]

                    classification_result = attack_log[attack_log['name'] == log_name]['pr'].values[0]

                    classification_result = "{:.4f}".format(classification_result)

                    axs[row, col].set_title(f'Prediction:{classification_result}')
                    axs[row, col].axis('off')  # Hide the axis

            
            fig.suptitle(self.orig_label_name + "_" + attack, fontsize=16)

            plt.subplots_adjust(wspace=0, hspace=0)
            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
            # Save the figure
            plt.savefig(self.image_dir + "plot_" + self.orig_label_name + "_" + attack + ".jpg", bbox_inches='tight')
           
    def top_plot(self, train_test = 'both', object_list = ['lemon','tractor','baseball','dining_table','shovel'], debug=False):

        attacks_name = ["fgsm" , "iterative", "universal", "clean_images"]
        columns = [
            'Attack',
            'Epsilon',
            'Top1',
            'Top5',
            'Count'
        ]

        df = pd.DataFrame(columns=columns)

        multiplied_epsilons = [x * 1000 for x in self.epsilons]
        combinations = list(product(attacks_name, multiplied_epsilons))
        df = pd.DataFrame(combinations, columns=['Attack', 'Epsilon'])
        df['Top1'] = 0.00
        df['Top5'] = 0.00
        df['Count'] = 0.00
        
        if debug:
            print(df)

        dataframes_list = []
        for object in object_list:
            data = pd.read_csv(self.image_dir + 'results_' + object + '.csv')
            dataframes_list.append(data)

        combined_results = pd.concat(dataframes_list, ignore_index=True)

        #print(combined_results)

        for index, row in combined_results.iterrows():
            if(train_test == row['name'].split('_')[2] or train_test == 'both'):
                if debug:
                    print(row['name'].split('_')[1])
                name = row['name'].split('_')[1]
                if debug:    
                    print(row['name'].split('_')[4][3:])
                eps = int(row['name'].split('_')[4][3:])
                
                if debug:
                    print(index)
                    print(row)

                if row['top1'] == True:
                    df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps), 'Top1'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps), 'Top1']
                if row['top5'] == True:
                    df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps), 'Top5'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps), 'Top5']

                df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps), 'Count'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps), 'Count']
            
        if debug:
            print(df)

        df.loc[(df['Attack'] == "clean_images"), "Top1"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Top1'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Top5"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Top5'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Count"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Count'].values[0]

        #df = df[df['Epsilon'] != 0.0]
        df['Epsilon'] = df['Epsilon'] / 10
        df['Top1'] = df['Top1'] / df['Count']
        df['Top5'] = df['Top5'] / df['Count']
        if debug:
            print(df)

        # Plot settings
        plt.figure(figsize=(14, 5))

        # Plot Top-1 Accuracy
        plt.subplot(1, 2, 1)
        for attack in df['Attack'].unique():
            subset = df[df['Attack'] == attack]
            plt.plot(subset['Epsilon'], subset['Top1'], marker='o', label=attack)
        # plt.title('Top-1 accuracy')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel('Top-1 accuracy')
        plt.legend()

        # Plot Top-5 Accuracy
        plt.subplot(1, 2, 2)
        for attack in df['Attack'].unique():
            subset = df[df['Attack'] == attack]
            plt.plot(subset['Epsilon'], subset['Top5'], marker='o', label=attack)
        # plt.title('Top-5 accuracy')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel('Top-5 accuracy')
        plt.legend()

        # Save the plots
        plt.tight_layout()
        plt.savefig(self.image_dir + "plot_top_"+train_test+".jpg", bbox_inches='tight') 

    def top_plot_loop(self, train_test = 'both', object_list = ['lemon','tractor','baseball','dining_table','shovel'], iter = 1, debug=False):

        attacks_name = ["fgsm", "iterative", "universal", "clean_images"]
        columns = [
            'Attack',
            'Epsilon',
            'Iter',
            'Top1',
            'Top5',
            'Count'
        ]

        df = pd.DataFrame(columns=columns)
        iter_list = [str(i) for i in range(1, iter+1, 2)]
        multiplied_epsilons = [x * 1000 for x in self.epsilons]
        combinations = list(product(attacks_name, multiplied_epsilons, iter_list))
        df = pd.DataFrame(combinations, columns=['Attack', 'Epsilon', 'Iter'])
        df['Top1'] = 0.00
        df['Top5'] = 0.00
        df['Count'] = 0.00
        df.loc[(df['Attack'] == "iterative"), "Iter"] = "11"
        
        if debug:
            print(df)

        dataframes_list = []
        for object in object_list:
            data = pd.read_csv(self.image_dir + 'results_' + object + '.csv')
            dataframes_list.append(data)

        combined_results = pd.concat(dataframes_list, ignore_index=True)

        #print(combined_results)

        for index, row in combined_results.iterrows():
            _name = row['name'].split('_')
            if(_name[1] == "table"):
                concatenated_string = _name[0] + _name[1]
                _name = [concatenated_string] + _name[2:]
            if(_name[1] == "iterative"):
                _name.pop(2)
            if(_name[1] == "fgsm"):
                _name.insert(2, 'n001')
            
            if(train_test == _name[3] or train_test == 'both'):
                name = _name[1]

                eps = int(_name[5][3:])

                n = int(_name[2][1:])    

                if row['top1'] == True:
                    df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top1'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top1']
                if row['top5'] == True:
                    df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top5'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top5']

            
                df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Count'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Count']
            
        if debug:
            print(df)

        df.loc[(df['Attack'] == "clean_images"), "Top1"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Top1'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Top5"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Top5'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Count"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Count'].values[0]

        #df = df[df['Epsilon'] != 0.0]
        df['Epsilon'] = df['Epsilon'] / 10
        df['Top1'] = df['Top1'] / df['Count']
        df['Top5'] = df['Top5'] / df['Count']

        print(df)

        df.loc[(df['Attack'] == "fgsm"), "Attack"] = "FGSM"
        df.loc[(df['Attack'] == "iterative"), "Attack"] = "BIM"
        df.loc[(df['Attack'] == "universal"), "Attack"] = "Universal"
        df.loc[(df['Attack'] == "clean_images"), "Attack"] = "Clean images"
        # Plot settings
        plt.figure(figsize=(14, 5))

        # Plot Top-1 Accuracy
        plt.subplot(1, 2, 1)
        for attack in df['Attack'].unique():
            subset = df[df['Attack'] == attack]
            plt.plot(subset['Epsilon'], subset['Top1'], marker='o', label=attack)
        # plt.title('Top-1 accuracy')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel('Top-1 accuracy')
        plt.legend(loc='center right')

        # Plot Top-5 Accuracy
        plt.subplot(1, 2, 2)
        for attack in df['Attack'].unique():
            subset = df[df['Attack'] == attack]
            plt.plot(subset['Epsilon'], subset['Top5'], marker='o', label=attack)
        # plt.title('Top-5 accuracy')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel('Top-5 accuracy')
        plt.legend()

        # Save the plots
        plt.tight_layout()
        plt.savefig(self.image_dir + "plot_top15_"+train_test+".jpg", bbox_inches='tight') 

    def chart(self, path):
        image_paths = ['ch_images_angle_5.jpg', 
                       'eps0005.jpg', 
                       'eps0050.jpg', 
                       'eps0100.jpg',
                       'eps0150.jpg',
                       'eps0300.jpg']
        epsilon_values = [0, 0.5, 5, 10, 15, 30]  # The epsilon values for each image

        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))  # Adjust nrows and ncols based on your number of images

        for ax, img_path, epsilon in zip(axs.flatten(), image_paths, epsilon_values):
            if(epsilon==0):
                img = mpimg.imread(self.image_dir + img_path)  # Read the image from the path
            else:
                img = mpimg.imread(self.image_dir + path  + img_path)  # Read the image from the path
            ax.imshow(img)                # Display the image
            title = f'$\epsilon = {epsilon}$' if epsilon > 0 else 'clean image'
            ax.set_title(title)           # Set the title for each subplot
            ax.axis('off')                # Turn off axis

        plt.tight_layout()
        plt.savefig(self.image_dir + "plot_" +path+ ".jpg", bbox_inches='tight') 

    def chart_method(self):
        image_paths = ['ch_images_angle_3.jpg', 
                       'adv_lemon_fgsm_test_img2_eps0150.jpg', 
                       'adv_lemon_universal_n001_test_img2_eps0150.jpg', 
                       'adv_lemon_iterative_attack_n011_test_img2_eps0150.jpg']
        titles = ["Clean Image", r"FGSM - $\epsilon$:15", r"Universal - $\epsilon$:15", r"BIM - $\epsilon$:15"]

        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))  # Adjust nrows and ncols based on your number of images

        for ax, img_path, title in zip(axs.flatten(), image_paths, titles):

            img = mpimg.imread(self.image_dir + img_path)  # Read the image from the path

            ax.imshow(img)                # Display the image
            ax.set_title(title)           # Set the title for each subplot
            ax.axis('off')                # Turn off axis

        plt.tight_layout()
        plt.savefig(self.image_dir + "plot_comparison.jpg", bbox_inches='tight') 

    def chart_accuracy(self, train_test = 'both', object_list = ['lemon','tractor','baseball','dining_table','shovel'], iter = 1, debug=False):

        attacks_name = ["fgsm", "iterative", "universal", "clean_images"]
        columns = [
            'Attack',
            'Epsilon',
            'Iter',
            'Top1',
            'Top5',
            'Accuracy',
            'Count'
        ]

        df = pd.DataFrame(columns=columns)
        iter_list = [str(i) for i in range(1, iter+1, 2)]
        multiplied_epsilons = [x * 1000 for x in self.epsilons]
        combinations = list(product(attacks_name, multiplied_epsilons, iter_list))
        df = pd.DataFrame(combinations, columns=['Attack', 'Epsilon', 'Iter'])
        df['Top1'] = 0.00
        df['Top5'] = 0.00
        df['Accuracy'] = 0.00
        df['Count'] = 0.00
        df.loc[(df['Attack'] == "iterative"), "Iter"] = "11"
        
        if debug:
            print(df)

        dataframes_list = []
        for object in object_list:
            data = pd.read_csv(self.image_dir + 'results_' + object + '.csv')
            dataframes_list.append(data)

        combined_results = pd.concat(dataframes_list, ignore_index=True)

        #print(combined_results)

        for index, row in combined_results.iterrows():
            _name = row['name'].split('_')
            if(_name[1] == "table"):
                concatenated_string = _name[0] + _name[1]
                _name = [concatenated_string] + _name[2:]
            if(_name[1] == "iterative"):
                _name.pop(2)
            if(_name[1] == "fgsm"):
                _name.insert(2, 'n001')
            
            if(train_test == _name[3] or train_test == 'both'):
                name = _name[1]

                eps = int(_name[5][3:])

                n = int(_name[2][1:])    

                if row['top1'] == True:
                    df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top1'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top1']
                if row['top5'] == True:
                    df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top5'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Top5']

            
                df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Count'] = 1 + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Count']
                df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Accuracy'] = float(row['pr']) + df.loc[(df['Attack'] == name) & (df['Epsilon'] == eps) & (df['Iter'] == str(n)), 'Accuracy']
            
        if debug:
            print(df)

        df.loc[(df['Attack'] == "clean_images"), "Top1"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Top1'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Top5"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Top5'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Count"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Count'].values[0]
        df.loc[(df['Attack'] == "clean_images"), "Accuracy"] = df.loc[(df['Attack'] == "universal") & (df['Epsilon'] == 0.0), 'Accuracy'].values[0]

        #df = df[df['Epsilon'] != 0.0]
        df['Epsilon'] = df['Epsilon'] / 10
        df['Top1'] = df['Top1'] / df['Count']
        df['Top5'] = df['Top5'] / df['Count']
        df['Accuracy'] = df['Accuracy'] / df['Count']

        print(df)

        print(df[['Attack','Epsilon','Accuracy']])

        df.loc[(df['Attack'] == "fgsm"), "Attack"] = "FGSM"
        df.loc[(df['Attack'] == "iterative"), "Attack"] = "BIM"
        df.loc[(df['Attack'] == "universal"), "Attack"] = "Universal"
        df.loc[(df['Attack'] == "clean_images"), "Attack"] = "Clean images"
        # Plot settings
        plt.figure(figsize=(14, 5))

        # Plot Top-1 Accuracy
        plt.subplot(1, 2, 1)
        for attack in df['Attack'].unique():
            subset = df[df['Attack'] == attack]
            plt.plot(subset['Epsilon'], subset['Top1'], marker='o', label=attack)
        # plt.title('Top-1 accuracy')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel('Top-1 accuracy')
        plt.legend(loc='center right')

        # Plot Top-5 Accuracy
        plt.subplot(1, 2, 2)
        for attack in df['Attack'].unique():
            subset = df[df['Attack'] == attack]
            plt.plot(subset['Epsilon'], subset['Top5'], marker='o', label=attack)
        # plt.title('Top-5 accuracy')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel('Top-5 accuracy')
        plt.legend()

        # Save the plots
        plt.tight_layout()
        plt.savefig(self.image_dir + "plot_top15_"+train_test+".jpg", bbox_inches='tight') 

    
        
if __name__ == "__main__":
    from chart_util import Chart_Util

    chart = Chart_Util(image_dir = "ch_images/")
    # chart.top_plot(object_list = ['lemon'], train_test = 'both')
    #chart.top_plot(object_list = ['lemon'], train_test = 'train', debug=True)
    # chart.top_plot(object_list = ['lemon'], train_test = 'test')
    
    #chart.top_plot_loop(train_test="train")
    #chart.top_plot_loop(train_test="test")
    #chart.chart(path = "adv_baseball_fgsm_test_img2_")
    #chart.chart(path = "adv_baseball_universal_n001_test_img2_")
    #chart.chart(path = "adv_baseball_iterative_attack_n011_test_img2_")

    #chart.chart_method()
    chart.chart_accuracy(train_test = "test")