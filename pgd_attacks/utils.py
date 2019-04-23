from copy import copy
import matplotlib.pyplot as plt
import os
import numpy as np

def deprocess_inception(y):
    x = copy(y).astype(np.float)
    x += 1.
    x /= 2.
    #x *= 255.
    return x

def preprocess_inception(y):
    x = copy(y).astype(np.float)
    x /= 255.
    x *= 2.
    x -= 1.
    return x

def calc_pert(img_num, x_true, x_adv):
    return deprocess_inception(x_adv[img_num] - x_true[img_num])

def display_adv_examples(imgNum, X_test_adv,
                         model, X_test, y_test,
                         verbose = False, X_test_bb = None):
    if verbose:
        print('Truth: ' + str(y_test[imgNum:imgNum+1]) )
        print('Model predict on true image: ' + str(model.predict(X_test[imgNum:imgNum+1])))
        print('Model predict on adversarial image: ' + str(model.predict(X_test_adv[imgNum:imgNum+1])))
        if X_test_bb is not None:
            print('Model predict on BB adversarial image: ' + str(model.predict(X_test_bb[imgNum:imgNum+1])))
        print("")

    plt.figure()
    print("True Image Deprocessed:")
    img = plt.imshow(deprocess_inception(X_test[imgNum]))
    img.set_cmap('hot')
    plt.axis('off')
    plt.show()

    print("Adversarial Image Deprocessed:")
    plt.figure()
    img = plt.imshow(deprocess_inception(X_test_adv[imgNum]))
    img.set_cmap('hot')
    plt.axis('off')
    plt.show()

    print("Perturbation*20:")
    plt.figure()
    img = plt.imshow(deprocess_inception((X_test_adv[imgNum] - X_test[imgNum])*20))
    img.set_cmap('hot')
    plt.axis('off')
    plt.show()
    
def getResults(directory, save = False):
    X_test = np.load(os.path.join(directory, 'data/val_test_x_preprocess.npy'), mmap_mode = "r")
    y_test = np.load(os.path.join(directory, 'data/val_test_y.npy'))

    X_test_pgd = np.load(os.path.join(directory, 'data/pgd_eps02_WhiteBox.npy'), mmap_mode = "r")
    X_test_pgd_bb = np.load(os.path.join(directory, 'data/pgd_eps02_BlackBox.npy'), mmap_mode = "r")
    
    if save:
        preds_clean = model.predict(X_test)
        np.save('data/winning_model_preds.npy', preds_clean)
    
    return (X_test, y_test, X_test_pgd, X_test_pgd_bb)