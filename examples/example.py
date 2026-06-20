from nnf.models import Model

model = Model.load_model("saved_models/None.json")

model.summary()
print(model.loss)