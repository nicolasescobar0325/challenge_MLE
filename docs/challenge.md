# Operationalizing and Deploying the Model

## Part I

In order to operationalize the model, transcribe the `.ipynb` file into the `model.py` file:

- If you find any bug, fix it.
- The DS proposed a few models in the end. Choose the best model at your discretion, argue why. **It is not necessary to make improvements to the model.**
- Apply all the good programming practices that you consider necessary in this item.
- The model should pass the tests by running `make model-test`.

> **Note:**
> - **You cannot** remove or change the name or arguments of **provided** methods.
> - **You can** change/complete the implementation of the provided methods.
> - **You can** create the extra classes and methods you deem necessary.

### MLE Takeout:
- **BUG:** I fixed the preprocessing function to extract the `period_of_day` from flight time. The way it was defined left range corners uncovered, causing more than 1k null values.
- **POSSIBLE-BUG:** During the exploratory analysis, the Data Scientist defined `training_dataset = shuffle(...)`. Even though it is a good approach, the defined DataFrame is not used after the definition. I did not fix this bug.
- **POSSIBLE-BUG:** A lot of usable features were not included in the model, some from the raw data and others created with transforming functions. There is no hint from the Data Scientist of the reasons not to use these features that seem to have predictive power (as seen in the exploratory analysis plots). I did not fix this bug but created the transforming functions as part of the pipeline for future use.
- **SUGGESTION:** During the exploratory phase, the Data Scientist selected the top 10 most important features from XGBoost to reduce dataset dimensionality. Even though this may not be harmful, it may be problematic to assume that both a tree-boosted model and a logistic regression will algorithmically learn the same way from the features. Additionally, the feature importance method from the XGBoost library does not account for feature interactions, making it less reliable than other methods like SHAP. I didn’t address this since the challenge mentioned not to improve the model.
- **IMPROVEMENTS:** To run the code, I had to include/change some libraries in the requirements files.

---

## Part II

Deploy the model in an `API` with `FastAPI` using the `api.py` file.

- The `API` should pass the tests by running `make api-test`.

> **Note:** 
> - **You cannot** use another framework.

### MLE Takeout:
- I had to redefine testing API calls, as they were defined as `flights`, which breaks the Pydantic standard of `data`. Fixing this to take `flights` correctly with FastAPI required adding complexity to the service definition.

---

## Part III

Deploy the `API` in your favorite cloud provider (we recommend using GCP).

- Put the `API`'s URL in the `Makefile` (`line 26`).
- The `API` should pass the tests by running `make stress-test`.

> **Note:** 
> - **It is important that the API is deployed until we review the tests.**

### MLE Takeout:
- Even though I was able to finish the coding part on time and make the code pass the tests correctly, I had a problem with GCP to set my billing account. This issue blocked me, as it didn’t allow me to use the Cloud Build service.

---

## Part IV

We are looking for a proper `CI/CD` implementation for this development.

- Create a new folder called `.github` and copy the `workflows` folder we provided inside it.
- Complete both `ci.yml` and `cd.yml` (consider what you did in the previous parts).

### MLE Takeout:
- Done.
