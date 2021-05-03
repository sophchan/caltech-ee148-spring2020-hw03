'''
Code for questions 6 and 7
-------------------------
Question 6
This data comes from the model 'mnist_model1.pt'.
The average loss for the training set is calculated in and imported from the file: 
    AverageLossByEpoch.xlxs
The average loss for the validation set is calculated from the outputs. 

Question 7
To reproduce models, change the variable x on line 305 in main.py to the 
appropriate fraction and rerun the file with the command 
    python main.py --batch-size 32 --epochs 10
Models are in the folder 'models' named by the fraction of data it is trained on:
    mnist_model1, mnist_model2, mnist_model4, mnist_model8, mnist_model16
for 1, 1/2, 1/4, 1/8, 1/16th of the training data. The remaining fraction is 
allocated towards the validation set. 
'''

import matplotlib.pyplot as plt 

# Question 6
epoch = list(range(1, 11))
train_loss_epoch = [0.9810, 0.5043, 0.3998, 0.3693, 0.3370, 0.3247, 0.3223, 0.2945, 0.2978, 0.3003]
val_loss_epoch = [0.3238,0.2725,0.2310,0.2287,0.1997,0.1800,0.1868,0.1760,0.1660,0.1663]

plt.plot(epoch, train_loss_epoch, 'C0', epoch, val_loss_epoch, 'C1')
plt.legend(['train', 'validation'])
plt.title('Average Training and Validation Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Error')
plt.show()

# Question 7
data_subset = [51005, 25505, 12755, 6380, 3192]
val_loss_subset = [8560/8995, 32503/34495, 43607/47245, 46580/53620, 46500/56808]
test_loss_subset = [9615/10000, 9543/10000, 9413/10000, 8890/10000, 8484/10000]

fig2, ax2 = plt.subplots()
ax2.loglog(data_subset, val_loss_subset, 'C1', data_subset, test_loss_subset, 'C2')
ax2.set_xticks([5000, 10000, 25000, 50000])
ax2.set_xticklabels([5000, 10000, 25000, 50000]) 
plt.legend(['validation', 'test'])
plt.title('Validation and Testing Accuracy vs Data for Training')
plt.xlabel('Data for Training (log)')
plt.ylabel('Accuracy (log)')
plt.show()