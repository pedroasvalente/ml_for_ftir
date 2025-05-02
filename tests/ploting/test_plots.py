import numpy as np

from ml4fir.ploting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_wavenumber_importances,
)
from ml4fir.config import TESTS_DIR

def test_plot_confusion_matrix():
    npz_path = os.path.join(TESTS_DIR, "ploting", "test_data", "plot_confusion_matrix.npz")
    npz = np.load(npz_path, allow_pickle=True)
    npz_dict = dict(npz)
    for key in [   'accuracy_score',
        'sample_type', 'train_percentage', 'test_name', 'target_name',
        'threshold', 'group_fam_to_use']:
        npz_dict[key] = npz_dict[key].item()
    npz_dict['accuracy_score'] = 0.75

    plot_confusion_matrix(plot_filepath=npz_path.replace(".npz", ".png"), **npz_dict)


def test_plot_roc_curve():
    npz_path = os.path.join(TESTS_DIR, "ploting", "test_data", "plot_roc_curve.npz")
    npz = np.load(npz_path, allow_pickle=True)
    npz_dict = dict(npz)
    for key in [ 'sample_type', 'train_percentage', 'test_name', 'target_name',
        'threshold', 'group_fam_to_use' ,"test_accuracy"]:
        npz_dict[key] = npz_dict[key].item()
    npz_dict['test_accuracy'] = 0.75

    plot_roc_curve(plot_filepath=npz_path.replace(".npz", ".png"), **npz_dict)


def test_plot_wavenumber_importances():
    npz_path = os.path.join(TESTS_DIR, "ploting", "test_data", "plot_wavenumber_importances.npz")
    npz = np.load(npz_path, allow_pickle=True)
    npz_dict = dict(npz)
    for key in [ 'sample_type', 'train_percentage', 'test_name',
    'target_name',
        "group_suffix",
        ]:
        npz_dict[key] = npz_dict[key].item()

    plot_wavenumber_importances(plot_filepath=npz_path.replace(".npz", ".png"), **npz_dict)

test_plot_confusion_matrix()
test_plot_roc_curve()
test_plot_wavenumber_importances()
