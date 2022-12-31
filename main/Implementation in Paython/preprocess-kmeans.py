import os
import cv2
import numpy as np
from concurrent import futures
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from joblib import dump

SOURCE = "./Dataset"
DEST = "./Dataset_modified"

TRAIN_FILES = sorted(os.listdir(os.path.join(SOURCE, "Train")),
                     key=lambda x: int(x[:-4]))
TRAIN_FILES_PATH = [os.path.join(SOURCE, "Train", file)
                    for file in TRAIN_FILES]

TEST_FILES = sorted(os.listdir(os.path.join(SOURCE, "Test")),
                    key=lambda x: int(x[:-4]))
TEST_FILES_PATH = [os.path.join(SOURCE, "Test", file) for file in TEST_FILES]

train_pics_np = np.zeros((len(TRAIN_FILES), 256, 256, 3), dtype=np.uint8)
train_masks_np = np.zeros((len(TRAIN_FILES), 256, 256, 3), dtype=np.uint8)
train_labels_np = np.zeros((len(TRAIN_FILES), 256, 256, 1), dtype=np.uint8)
test_pics_np = np.zeros((len(TEST_FILES), 256, 256, 3), dtype=np.uint8)
test_masks_np = np.zeros((len(TEST_FILES), 256, 256, 3), dtype=np.uint8)
test_labels_np = np.zeros((len(TEST_FILES), 256, 256, 1), dtype=np.uint8)


def find_centers(file_paths: list[str]) -> MiniBatchKMeans:
    clf = MiniBatchKMeans(batch_size=256,
                          n_clusters=15, tol=1e-4, verbose=1, max_iter=1000, random_state=1)
    masks = np.zeros((len(file_paths), 256, 256, 3))
    for i, fp in enumerate(file_paths):
        img = cv2.cvtColor(
            cv2.imread(fp), cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        masks[i] = img[:, 256:, :]

    for i in range(len(masks)//50):
        clf.partial_fit(masks[i*50:(i+1)*50].reshape(-1, 3))
    return clf


def process_train(i, path, centers):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
    pic = img[:, :256, :]
    mask = img[:, 256:, :]
    label = np.array([
        np.argmin(np.linalg.norm(x-centers, axis=1))
        for x in mask.reshape(-1, 3)
    ])
    train_pics_np[i] = pic
    train_masks_np[i] = mask
    train_labels_np[i] = label.reshape(256, 256, 1)

    # cv2.imwrite(os.path.join(DEST, "Train", "pic", name), pic),
    # cv2.imwrite(os.path.join(DEST, "Train", "mask", name), mask),
    # cv2.imwrite(os.path.join(DEST, "Train", "label",
    #             name), label.reshape(256, 256, 1)),


def process_test(i, path, centers):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
    pic = img[:, :256, :]
    mask = img[:, 256:, :]
    label = np.array([
        np.argmin(np.linalg.norm(x-centers, axis=1))
        for x in mask.reshape(-1, 3)
    ])
    # cv2.imwrite(os.path.join(DEST, "Test", "pic", name), pic),
    # cv2.imwrite(os.path.join(DEST, "Test", "mask", name), mask),
    # cv2.imwrite(os.path.join(DEST, "Test", "label",
    #             name), label.reshape(256, 256, 1)),

    test_pics_np[i] = pic
    test_masks_np[i] = mask
    test_labels_np[i] = label.reshape(256, 256, 1)


def run(clf: MiniBatchKMeans):
    for i, path in tqdm(enumerate(TRAIN_FILES_PATH)):
        process_train(i, path, clf.cluster_centers_)
    # with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     executor.map(process_train, list(range(len(TRAIN_FILES))),
    #                  TRAIN_FILES_PATH)
    np.save("Dataset_modified/Train/pics-kmeans.npy",
            train_pics_np, allow_pickle=False)
    np.save("Dataset_modified/Train/masks-kmeans.npy",
            train_masks_np, allow_pickle=False)
    np.save("Dataset_modified/Train/labels-kmeans.npy",
            train_labels_np, allow_pickle=False)

    for i, path in tqdm(enumerate(TEST_FILES_PATH)):
        process_test(i, path, clf.cluster_centers_)
    # with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     executor.map(process_test, list(range(len(TEST_FILES))),
    #                  TEST_FILES_PATH)
    np.save("Dataset_modified/Test/pics-kmeans.npy",
            test_pics_np, allow_pickle=False)
    np.save("Dataset_modified/Test/masks-kmeans.npy",
            test_masks_np, allow_pickle=False)
    np.save("Dataset_modified/Test/labels-kmeans.npy",
            test_labels_np, allow_pickle=False)


if __name__ == "__main__":
    clf = find_centers(TRAIN_FILES_PATH)
    print(clf.cluster_centers_)
    dump(clf, "kmeans-clf.joblib")

    run(clf)
