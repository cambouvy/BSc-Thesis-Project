import os
import shutil
import tempfile
import copy
import json

from IPython import embed
import pytest

from qlknn.pipeline.pipeline import *


@pytest.fixture
def train_nn_case(default_settings, tmpdir):
    settings = copy.deepcopy(default_settings)
    copy.deepcopy(settings)
    settings.pop("train_dims")

    train_nn_job = TrainNN(settings=settings, train_dims=["efiITG_GB"], uid="test")
    train_nn_job.interact_with_nndb = False
    return tmpdir, settings, train_nn_job


class TestDummyTask:
    def test_create(self):
        task = DummyTask()

    def test_run(self):
        task = DummyTask()
        task.run()


class TestTrainNN:
    def test_launch_train_NN(self, train_nn_case):
        jobdir, settings, train_nn_job = train_nn_case
        settings["train_dims"] = train_nn_job.train_dims
        with jobdir.as_cwd():
            with open("settings.json", "w") as file_:
                json.dump(settings, file_)
            train_nn_job.launch_train_NDNN()

    def test_run(self, train_nn_case):
        jobdir, settings, train_nn_job = train_nn_case
        settings["train_dims"] = train_nn_job.train_dims
        with jobdir.as_cwd():
            with open("settings.json", "w") as file_:
                json.dump(settings, file_)
            train_nn_job.sleep_time = 0
            train_nn_job.run()
