import os
import cv2
import numpy as np
from concurrent import futures
from tqdm import tqdm


SOURCE = "./Dataset"
DEST = "./Dataset_modified"

TRAIN_FILES = sorted(os.listdir(os.path.join(SOURCE, "Train")),
                     key=lambda x: int(x[:-4]))
TRAIN_FILES_PATH = [os.path.join(SOURCE, "Train", file)
                    for file in TRAIN_FILES]

TEST_FILES = sorted(os.listdir(os.path.join(SOURCE, "Test")),
                    key=lambda x: int(x[:-4]))
TEST_FILES_PATH = [os.path.join(SOURCE, "Test", file) for file in TEST_FILES]

colors = np.array([
    [0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
], dtype=np.uint8)


train_pics_np = np.zeros((len(TRAIN_FILES), 256, 256, 3), dtype=np.uint8)
train_masks_np = np.zeros((len(TRAIN_FILES), 256, 256, 3), dtype=np.uint8)
train_labels_np = np.zeros((len(TRAIN_FILES), 256, 256, 1), dtype=np.uint8)
test_pics_np = np.zeros((len(TEST_FILES), 256, 256, 3), dtype=np.uint8)
test_masks_np = np.zeros((len(TEST_FILES), 256, 256, 3), dtype=np.uint8)
test_labels_np = np.zeros((len(TEST_FILES), 256, 256, 1), dtype=np.uint8)


def process_train(i, path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
    pic = img[:, :256, :]
    mask = img[:, 256:, :]
    label = np.array([
        np.argmin(np.linalg.norm(x-colors, axis=1))
        for x in mask.reshape(-1, 3)
    ])
    train_pics_np[i] = pic
    train_masks_np[i] = mask
    train_labels_np[i] = label.reshape(256, 256, 1)

    # cv2.imwrite(os.path.join(DEST, "Train", "pic", name), pic),
    # cv2.imwrite(os.path.join(DEST, "Train", "mask", name), mask),
    # cv2.imwrite(os.path.join(DEST, "Train", "label",
    #             name), label.reshape(256, 256, 1)),


def process_test(i, path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
    pic = img[:, :256, :]
    mask = img[:, 256:, :]
    label = np.array([
        np.argmin(np.linalg.norm(x-colors, axis=1))
        for x in mask.reshape(-1, 3)
    ])
    # cv2.imwrite(os.path.join(DEST, "Test", "pic", name), pic),
    # cv2.imwrite(os.path.join(DEST, "Test", "mask", name), mask),
    # cv2.imwrite(os.path.join(DEST, "Test", "label",
    #             name), label.reshape(256, 256, 1)),

    test_pics_np[i] = pic
    test_masks_np[i] = mask
    test_labels_np[i] = label.reshape(256, 256, 1)


def run():
    # for i, path in tqdm(enumerate(TRAIN_FILES_PATH)):
    #     process_train(i, path)
    # # with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    # #     executor.map(process_train, list(range(len(TRAIN_FILES))),
    # #                  TRAIN_FILES_PATH)
    # np.save("Dataset_modified/Train/pics.npy", train_pics_np, allow_pickle=False)
    # np.save("Dataset_modified/Train/masks.npy", train_masks_np, allow_pickle=False)
    # np.save("Dataset_modified/Train/labels.npy", train_labels_np, allow_pickle=False)

    for i, path in tqdm(enumerate(TEST_FILES_PATH)):
        process_test(i, path)
    # with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     executor.map(process_test, list(range(len(TEST_FILES))),
    #                  TEST_FILES_PATH)
    np.save("Dataset_modified/Test/pics.npy", test_pics_np, allow_pickle=False)
    np.save("Dataset_modified/Test/masks.npy", test_masks_np, allow_pickle=False)
    np.save("Dataset_modified/Test/labels.npy", test_labels_np, allow_pickle=False)


if __name__ == "__main__":
    run()
