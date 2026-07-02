# Backs the README demo GIF (see pipeline-demo.tape) with realistic, instant output
# instead of hitting a real server/GPU. Call signatures and print idioms mirror the
# real API and the notebooks/ tutorials. Regenerate with: vhs assets/pipeline-demo.tape
import time


def step(title):
    print("\033[H\033[2J", end="")
    print(title)
    print("─" * len(title))
    print()


def finale():
    print("\033[H\033[2J", end="")
    print("🏁  From raw data to a deployed model in 7 lines")
    print()
    print("✅ Dataset loaded — any format, auto-detected")
    print("✅ Patient-wise split — no data leakage")
    print("✅ Trained & tracked in MLflow")
    print("✅ Validated before shipping")
    print("✅ Deployed and live")


def _bar(pct, width=22):
    filled = int(width * pct)
    return "━" * filled + "─" * (width - filled)


class SplitResult(dict):
    def save(self):
        time.sleep(0.3)


class ImageDataset:
    def __init__(self, project, n=780):
        self.project = project
        self._n = n

    def __len__(self):
        return self._n

    def split(self, *, by_patient=False, seed=None, **ratios):
        time.sleep(0.4)
        return SplitResult(train=range(546), val=range(117), test=range(117))


def build_dataset(project):
    time.sleep(0.4)
    return ImageDataset(project)


class UNetPPTrainer:
    def __init__(self, project, **kwargs):
        self.project = project

    def fit(self):
        for epoch in range(1, 6):
            pct = epoch / 5
            print(f"\rEpoch {epoch}/5 {_bar(pct)} loss={0.51 - 0.08 * epoch:.3f}", end="", flush=True)
            time.sleep(0.25)
        print()
        return {"model": object(), "test_results": [{"dice": 0.8912, "iou": 0.8054}]}


def validate_model(model, dataset=None, **kwargs):
    time.sleep(0.5)
    return "[v] task_type matches project\n[v] inference smoke test passed"


class DeployJob:
    status = "running"


class _Deploy:
    def start(self, **kwargs):
        time.sleep(0.5)
        return DeployJob()


class _Api:
    deploy = _Deploy()


api = _Api()
