import cupy as cp
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from Checkpoint import Save_model, restore_lr
from Visualize_result import *
from VOCDataset import *
from Loss import *

class Compose:
    def __init__(self, transforms):
        self.transform = transforms

    def __call__(self, boxes, image):
        for t in self.transform:
            boxes, image = boxes,t(image)
        return boxes, image

def Prepare_data():
    image_dir = r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\Project\DL\Convolutional network\Obj_Detection\YOLOv1\images'
    label_dir = r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\Project\DL\Convolutional network\Obj_Detection\YOLOv1\labels'
    transforms = Compose([transform.Resize((224, 224)), transform.ToTensor()])

    train_dataset = VOCDataset(cvs_file = '100examples.csv', label_dir = label_dir, image_dir = image_dir, transforms = transforms)

    test_dataset_org = VOCDataset_org_data(cvs_file = '8examples.csv', label_dir = label_dir, image_dir = image_dir)

    test_dataset = VOCDataset(cvs_file = '8examples.csv', label_dir = label_dir, image_dir = image_dir, transforms = transforms)

    test_labels_org = []
    test_img_org = []
    for i in range(test_dataset_org.__len__()):
        img_org, label = test_dataset_org.__getitem__(i)
        test_labels_org.append(label)
        test_img_org.append(img_org)

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers = 2,
                              pin_memory = True,
                              drop_last = True)

    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers = 2,
                             pin_memory = True,
                             drop_last = True)

    return train_loader, test_loader, test_labels_org, test_img_org

def Train_model(model, loader, epochs):
    loss = 0
    for epoch in range(epochs):
        loss_compute = Loss()
        running_loss = 0
        pbar = tqdm(loader)
        num_of_batches = len(loader)
        total_img = 0
        for batch_idx, (X_loader, y_loader) in enumerate(pbar):
            X_numpy = X_loader.numpy().transpose(0, 2, 3, 1)
            X_numpy = cp.asarray(X_numpy).astype(cp.float32)
            Y_numpy = cp.asarray(y_loader.numpy()).astype(cp.float32)
            total_img += X_numpy.shape[0]

            y_hat = model.forward(X_numpy, training=True)
            loss = loss_compute.forward(y_hat.get(), Y_numpy.get())
            running_loss += float(loss)

            dZ_init = loss_compute.backward()
            if isinstance(dZ_init, torch.Tensor):
                dZ_init = cp.asarray(dZ_init.cpu().numpy()).astype(cp.float32)

            print(f"\ny_hat min={float(y_hat.min()):.3f}, max={float(y_hat.max()):.3f}")
            print(f"dZ    min={float(dZ_init.min()):.3f}, max={float(dZ_init.max()):.3f}")

            global_step = batch_idx + num_of_batches * epoch
            model.backward(dZ_init, global_step, epoch)

            pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss):.4f}")
            del X_numpy, Y_numpy, y_hat, loss
            cp.get_default_memory_pool().free_all_blocks()

        loss = running_loss / total_img
        print(f"\nKẾT THÚC EPOCH {epoch}: Loss/Image: {loss:.4f}")

    save_name = "Parameter_cache.pkl"
    Save_model(model, save_name)
    return loss

def Valid_model(model, loader, y_org, test_img_org, plot_result_img = True):
    print("Validating model...")
    loss_compute = Loss()
    running_loss = 0
    pbar = tqdm(loader)
    total_img = 0
    pred = []

    for batch_idx, (X_loader, y_loader) in enumerate(pbar):
        X_numpy = X_loader.numpy().transpose(0, 2, 3, 1)
        X_numpy = cp.asarray(X_numpy).astype(cp.float32)
        Y_numpy = cp.asarray(y_loader.numpy()).astype(cp.float32)
        total_img += X_numpy.shape[0]

        y_hat = model.forward(X_numpy, training=True)
        loss = loss_compute.forward(y_hat.get(), Y_numpy.get())

        y_hat = non_max_suppression(y_hat)
        pred.extend(y_hat)

        running_loss += loss

        del X_numpy, Y_numpy, y_hat, loss
        cp.get_default_memory_pool().free_all_blocks()

    print(f'Total Loss: {float(running_loss):.4f}')
    loss = running_loss / total_img
    if plot_result_img == True:
        Visualization_img_result(pred, y_org, test_img_org)
    return float(loss)

def Test_model(model, loader, y_org, test_img_org):
    print("Testing model...")
    loss_compute = Loss()
    running_loss = 0
    pbar = tqdm(loader)
    total_img = 0
    pred = []

    for batch_idx, (X_loader, y_loader) in enumerate(pbar):
        X_numpy = X_loader.numpy().transpose(0, 2, 3, 1)
        X_numpy = cp.asarray(X_numpy).astype(cp.float32)
        Y_numpy = cp.asarray(y_loader.numpy()).astype(cp.float32)
        total_img += X_numpy.shape[0]

        y_hat = model.forward(X_numpy, training=True)
        loss = loss_compute.forward(y_hat.get(), Y_numpy.get())

        y_hat = non_max_suppression(y_hat)

        pred.extend(y_hat)

        running_loss += loss

        del X_numpy, Y_numpy, y_hat, loss
        cp.get_default_memory_pool().free_all_blocks()

    print(f'Total Loss: {float(running_loss):.4f}')
    loss = running_loss / total_img
    Visualization_img_result(pred, y_org, test_img_org)
    return loss

def Model_loss_tracking(model, train_loader, valid_loader, y_valid_org, test_img_org, epochs):
    training_loss = []
    validating_loss = []

    for epoch in range(epochs):
        training_loss.append(Train_model(model, train_loader, epochs=1))
        validating_loss.append(Valid_model(model, valid_loader, y_valid_org, test_img_org, plot_result_img=False))
        print(f'Complete epoch {epoch} / {epochs} !!! ')

        Visualization_train_valid_result(training_loss, validating_loss)

def Activate_model(model, type):
    train_loader, test_loader, label_org, test_img_org = Prepare_data()
    if type == "train":
        restore_lr(model, new_lr = 0.00001 * 0.95 ** 6)
        Train_model(model, train_loader, epochs = 5)
    elif type == "valid":
        Valid_model(model, test_loader, label_org, test_img_org, plot_result_img=True)
    elif type == "test":
        Test_model(model, test_loader, label_org, test_img_org)
    elif type == "model_loss_traking":
        Model_loss_tracking(model, train_loader, test_loader, label_org, test_img_org, epochs = 1)
