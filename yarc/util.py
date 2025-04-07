import numpy as np
import yarc
import matplotlib.pyplot as plt
from   matplotlib import colors

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_one(ax, i, aimg):
    input_matrix = aimg.data
    fs=12 
    ax.imshow(input_matrix, cmap=cmap, norm=norm)


    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    
    #ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs-2)

def plot_task(puzzle:yarc.Arcset, i:int, t:str):
    """    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app    """    
    fs=12    
    num_train = puzzle.shots
    num_test  = (len(puzzle.imgs)//2) - num_train
    
    w=num_train+num_test
    fig, axs  = plt.subplots(2, w, figsize=(2*w,2*2))
    #fig, axs  = plt.subplots(2, w, figsize=(1.5*w, 1.5*2))
    plt.suptitle(f'Arcset #{i}, {t}:', fontsize=fs, fontweight='bold', y=1)
    #plt.subplots_adjust(hspace = 0.15)
    #plt.subplots_adjust(wspace=20, hspace=20)
    
    for j in range(num_train):     
        plot_one(axs[0, j], j,puzzle.imgs[j*2])
        plot_one(axs[1, j], j,puzzle.imgs[j*2+1])
        
    for k in range(num_test):     
        plot_one(axs[0, num_train+k], k, puzzle.imgs[(num_train+k)*2])
        input_matrix = puzzle.imgs[(num_train+k)*2+1].data
    
        axs[1, num_train+k].imshow(input_matrix, cmap=cmap, norm=norm)
        axs[1, num_train+k].grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
        axs[1, num_train+k].set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
        axs[1, num_train+k].set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
        axs[1, num_train+k].set_xticklabels([])
        axs[1, num_train+k].set_yticklabels([])
        axs[1, num_train+k].set_title('Test output')
    
        axs[1, num_train+k] = plt.figure(1).add_subplot(111)
        axs[1, num_train+k].set_xlim([0, num_train+1])
        
        for m in range(1, num_train):
            axs[1, num_train+k].plot([m,m],[0,1],'--', linewidth=1, color = 'black')
        
        axs[1, num_train+k].plot([num_train,num_train],[0,1],'-', linewidth=3, color = 'black')
    
        axs[1, num_train+k].axis("off")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black') 
    fig.patch.set_facecolor('#dddddd')
   
    plt.tight_layout()
    
    print(f'#{i}, {t}') # for fast and convinience search
    plt.show()  
    
    print()
    #print()
    