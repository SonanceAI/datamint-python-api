# Regenerate the README demo GIF with: vhs assets/pipeline-demo.tape
# (run from the repo root; needs vhs: https://github.com/charmbracelet/vhs)
Output assets/pipeline-demo.gif

Set Shell "bash"
Set FontSize 17
Set Width 1000
Set Height 340
Set Theme "Dracula"
Set Padding 20
Set WindowBar Colorful
Set BorderRadius 8
Set TypingSpeed 35ms

Hide
Type `python3 -iq assets/pipeline_demo_mock.py`
Enter
Sleep 500ms
Type `step("📦  1/5  Build a dataset — no format handling needed")`
Enter
Sleep 300ms
Show

Type@55ms `dataset = build_dataset("A_Cool_Project")`
Enter
Sleep 1400ms
Type@55ms `print(f"{type(dataset).__name__}: {len(dataset)} resources")`
Enter
Sleep 2600ms

Hide
Type `step("✂️  2/5  Patient-wise split — no leakage between sets")`
Enter
Sleep 300ms
Show

Type `parts = dataset.split(train=0.7, val=0.15, test=0.15, by_patient=True, seed=42)`
Enter
Sleep 900ms
Type `print({name: len(ds) for name, ds in parts.items()})`
Enter
Sleep 900ms
Type `parts.save()`
Enter
Sleep 2200ms

Hide
Type `step("🧠  3/5  Train")`
Enter
Sleep 300ms
Show

Type `trainer = UNetPPTrainer(project="A_Cool_Project", max_epochs=5)`
Enter
Sleep 400ms
Type `results = trainer.fit()`
Enter
Sleep 2200ms
Type@55ms `metrics = results['test_results'][0]`
Enter
Sleep 900ms
Type@55ms `print(f"dice: {metrics['dice']:.4f}")`
Enter
Sleep 1000ms
Type@55ms `print(f"iou: {metrics['iou']:.4f}")`
Enter
Sleep 2600ms

Hide
Type `step("✅  4/5  Validate before shipping it")`
Enter
Sleep 300ms
Show

Type `report = validate_model(results["model"], dataset=dataset)`
Enter
Sleep 800ms
Type `print(report)`
Enter
Sleep 2200ms

Hide
Type `step("🚀  5/5  Deploy")`
Enter
Sleep 300ms
Show

Type `job = api.deploy.start(model_name="A_Cool_Project", model_alias="champion")`
Enter
Sleep 800ms
Type `print(f"Status: {job.status}")`
Enter
Sleep 1800ms

Hide
Type `finale()`
Enter
Show
Sleep 4500ms
