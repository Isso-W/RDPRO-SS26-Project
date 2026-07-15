# Jiaozi Project Report (for non-technical readers)

> This report is written for non-technical students such as product and operations students. Every technical term will be explained in vernacular when it appears for the first time.
> Every design choice has a clear "why". After reading, you should be able to explain clearly to others: What is Jiaozi?
> Why is it valuable and what is the difference from competing products?

> **About "done" or "to be verified"**: This report uses two markers to distinguish feature maturity:
> **✅ Integrated** (code and offline testing are located at the current `main`),
> **🔬 To be verified/under planning** (requires real data, GPU or further implementation).
> In this way, non-technical students can see at a glance which ones can be "demonstrated now" and which ones still need to be demonstrated.

---

## 1. Explain in one sentence what Jiaozi is

**Jiaozi is a "AI visual model selection consultant + automatic code generation tool". **

For example: you want to decorate a house (= train a AI model that can recognize pictures), but you don't know how to decorate. What Jiaozi does is:

1. **Understand your needs** ("I want to identify leaf diseases, be accurate, but I need to be able to run it on my mobile phone")
2. **Take a look at your house** (your picture data set - how many pictures, how many categories, how sharp)
3. **Select the most suitable complete solution from a decoration knowledge base** (Which AI model skeleton to use, what parts to use, and how to train)
4. **Generate runnable code directly** (generate a complete set of training codes that can be run directly)

The user only needs to give a sentence of requirements + a data set, and Jiaozi can output "what model should be used + why + runnable code".

> Glossary:
> - **Model/AI Model**: A program that can recognize pictures. It needs to be "trained" with data.
> - **Training**: Feed a bunch of annotated pictures to the model and let it learn to recognize.
> - **Dataset**: Your pile of annotated images.

---

## 2. Why is it needed (what pain points are solved)

Ordinary people/small and medium-sized teams who want to do image recognition AI will get stuck in three places:

| Pain points | Specific performance |
|---|---|
| **Don't know which model to choose** | There are dozens of AI models, each with its own strengths. Choosing the wrong one will be slow and inaccurate. |
| **I don't know how to configure parameters** | For the same model, if the parameters are mismatched (such as the "learning speed" setting is wrong), the results will be very different. |
| **Can't write training code** | Even if you choose the right one, you still need to write hundreds of lines of code to run it. |

Jiaozi **automates** these three things: model selection + parameter configuration + code generation, one-stop.

---

## 3. How does the whole thing work (four-step assembly line)

Jiaozi is internally divided into four modules, which are connected like an assembly line. Let's use the "decoration" analogy:

```
Your one sentence request ─┐
├─→ ① Understand the needs ─┐
Your data set ─┘ ├─→ ③ Model selector ─→ ④ Generate code (code)
② Look at the data ───┘
```

| module | Popular speaking | What to do |
|---|---|---|
| **Module 1: Understand the needs** | Translate human words into lists | Extracted from "accuracy, economy, and mobile phone access": task type, whether accuracy or speed is more important, and whether there are any special constraints |
| **Module 2: Look at the data** | Dataset profiling | How many pictures were counted, how many categories, clarity, and whether they were "partial" (a certain category was particularly numerous) |
| **Module 3: Model Selector** | Pick a solution from the knowledge base | Comprehensive requirements + data, select the most appropriate model + components |
| **Module 4: Code Generation** | Turn plans into code | Generate ready-to-run training/evaluation/prediction code |

Let's go through them one by one and explain the reasons for each design choice.

---

## 4. Detailed explanation module by module + why it is designed this way

### Module 1: Understand the needs

**What to do**: The user inputs a sentence of natural language ("Identify cassava leaf diseases, priority is accuracy, and the categories are unbalanced"). Module 1 uses a **large language model** (that is, ChatGPT, the kind of AI that can read and write) to translate it into a structured list:

- Task type (classification/detection/segmentation/feature extraction)
- Priority (valuing speed/accuracy/balance)
- Constraints (whether it should be real-time, whether it should be on edge devices, whether the categories are unbalanced...)
- **Evaluation indicators** (What to score: accuracy / AUC / F1 / grading consistency...)

**Why is it designed like this**:
- **Why use a large language model**: User needs are free human speech, and rule matching cannot handle "vague expressions". Large language models are best at transforming human speech into structures.
- **Why "Evaluation Index" identification was recently added**: In the past, the system defaulted to "accuracy" regardless of what the user said. But for some tasks (such as medicine according to AUC, disease classification according to "grading consistency") it is wrong to use accuracy. Now that the user says "Rate by AUC", the system will actually use AUC - otherwise the selected solution will be optimized for the wrong target.

### Module 2: Dataset profiling

**What to do**: Load the user's data set, statistics: total number of pictures, number of categories, how many pictures in each category (to determine whether it is biased), picture resolution (width and height), and whether it is color or grayscale.

**Why is it designed like this**:
- **Why look at data instead of just listening to needs**: Users often don't know key facts about their data. For example, "I have a lot of data" - but 20,000 pictures are divided into 200 categories, with only 100 pictures in each category, which is actually **small data**. The size of the data directly determines whether to use a large model or a small model, and whether to prevent overfitting.
- **An efficiency issue we have fixed**: During the physical examination, each picture would be decoded **repeatedly 3~4 times** (in order to count the size, color, and format separately). 9,000 pictures would be decoded more than 30,000 times, which takes several minutes. We changed it to **scan only once + sampling**, reducing it from a few minutes to a few seconds - **This does not affect the results, it is purely a simple optimization of an inefficient method. **

> Glossary:
> - **Overfitting**: The model has "memorized" the training images, but does not recognize the new images - it is most likely to occur when there is too little data.
> - **Large model vs Small model**: Large models are more accurate but slower and more expensive, and are easier to overfit on small data; small models are the opposite.

### Module 3: Model Selector (Core)

**What to do**: Comprehensive information from modules 1+2, select the most appropriate "complete solution" from a **knowledge base**: model skeleton + pre-training weights + output components + loss function + optimizer + training strategy.

**What does this knowledge base look like**: We have manually compiled 14 mainstream model skeletons, 22 ready-made "pre-training weights" ("semi-finished products" that have been trained by others on massive data and can be directly used for fine-tuning), and supporting components. The relationship between them (who can match whom, who is a substitute for whom) is also recorded in a "relationship diagram".

**Why is it designed like this**:
- **Why use artificial knowledge base instead of letting AI search on-site online**: The knowledge base is **controllable, explainable, and stable** - We can clearly explain "why we chose this", while online search is a black box and unstable. (This is the key difference between us and competing products, see Section 5.)
- **How ​​to choose (comprehensive scoring of multiple signals)**: Matching based on data size, matching based on "accuracy or speed", filtering based on constraints, and adding a "semantic similarity" (matching user description and model introduction). In the final comprehensive ranking, the top 3 will be selected.

**Three important abilities we added to it** (all with clear reasons):

1. **Constraint Awareness/Budget Awareness** ✅ Now online
   - **Question**: In the past, no matter you said "you want to use the mobile phone", the system would push a large model with hundreds of millions of parameters, which would not fit into the mobile phone at all.
   - **Method**: Each model is marked with "volume" (amount of parameters, calculation amount), and the user can give a **budget** (such as "no more than 12 million parameters"). The system will **automatically cut off the version that exceeds the budget and automatically downgrade it to a smaller version that can fit into the budget** (for example, if ResNet50 is too big → it will be automatically replaced with ResNet18).
   - **Why it matters**: Real deployment always has constraints (mobile phone memory, latency). A solution that is "the best within the constraints" is much more useful than a solution that has the highest naked score but cannot be deployed at all.

2. **Recommender that can accumulate and explain ("self-learning RAG")** ✅ Already integrated, the real gain needs to be verified
   - **Problem**: Relying solely on manual rule selection is not smart enough, and **won't learn from experience** - the same mistake will be made a hundred times.
   - **How ​​to do it**: After each training run, the system records "this data set looks like this + this solution was used + how many points it scored + how much it cost" into a **memory bank**. Next time you encounter a similar data set, give priority to recommending solutions that worked really well in the past, and give reasons ("Recommend it because it scored 0.86 on the similar beans data set").
   - **What to do with cold start** (the memory is empty at the beginning): We use a cheap method called **LogME** - without actual training, you can quickly estimate "which model skeleton is the best match" on your data, and use it as a starting signal.
   - **Why it matters**: This is the source of Jiaozi **The more you use it, the smarter it becomes**, and it is also the biggest difference between it and competing products (competing products start from scratch every time and are forgetful).

3. **The "hyperparameter" recipe layer will be recommended** ✅ Already connected to the current integration link (regular version)
   - **Noun**: Hyperparameters = settings during training (learning speed, number of training rounds, image resolution, etc.). If mismatched, the results will be poor.
   - **Problem**: In the past, these settings were all hard-coded fixed values, which were the same regardless of the model or data - very unsmart.
   - **Method**: Make a "recipe" and give appropriate settings according to (model type, training strategy, data size, image clarity, color mode). For example, "Fine-tune the large Transformer model → Use a very slow learning speed for the skeleton part, otherwise you will forget everything you learned in pre-training."
   - **Why are these rules reliable**: Many rules of hyperparameters are **mechanical and recognized by the industry** (not as guessing as "which model to choose"), so regularization is reliable.
   - **How ​​to connect to the pipeline now**: After selecting the candidate solution, module 3 will add the recipe and source description to the classification task; when module 4 generates code, if the user does not specify it manually, this recipe will be used.
   - **Future**: In the future, a large language model can be used to propose these settings (it has read massive training recipes), but it will be constrained by rule "guardrails" and will not get out of control.

### Module 4: Generate code (generate code)

**What to do**: Turn the solution selected in Module 3 into a set of codes that can be run directly: training, evaluation, prediction, running scripts, configuration files, etc.

**Why is it designed like this**:
- **Most of the code is generated using "templates"**: The training/evaluation skeleton is fixed and reliable, and templates are used to ensure that there will be no errors**.
- **Only the small part of "model structure" can be written by a large language model** (if the writing is not good, the template will be returned).
  - **Why is it divided like this**: Leave reliability to templates and flexibility to AI.
- **Our two most recent improvements**:
  - **Correct errors before giving up**: In the past, the code written by AI would be directly returned to the template as soon as an error was reported. Now I will tell AI the error and let it be corrected, and then return it after correcting it several times and it still doesn't work. This is smarter, but there is an upper limit on the number of times (to control costs), so it is not an unlimited toss.
  - **Compatible with new models**: Some new versions of AI (such as gpt-5.x) have special requirements for calling parameters. We have automatically adapted them and no longer report errors.

---

## 5. Our core differences/why we are valuable

There is a strong competing product on the market called **MLE-STAR** (produced by Google). To be honest, in terms of "naked score" (purely comparing whose trained model is more accurate), it is better than us - because it will repeatedly trial and error, constantly search, and combine multiple models (this is called ensemble, integration).

But the way it works dictates three **structural weaknesses**, and that's where Jiaozi is:

| Dimensions | MLE-STAR (competitive product) | Jiaozi |
|---|---|---|
| **cost** | Each task repeatedly calls AI + repeated training, **very expensive and slow** | Basically done in one go, **dozens of times cheaper** |
| **Explainable** | A black box gives you results but you can't tell why. | **Every recommendation has a reason** (data signals + historical evidence) |
| **Will it accumulate** | Each task starts from scratch and **forgets it after it is done** | **The more you use it, the more accurate it becomes** (memory library), the marginal cost approaches 0 |
| **Constraint Awareness** | Just pursue high scores, regardless of whether they can be deployed | **Choose the best within the budget**, naturally adapt to deployment |

> **Maturity Note**: In the above table, constraint awareness, result memory and interpretation path have all been integrated. Current evidence supports these
> The software path can run, but there are not enough real historical records to prove the effect of "the more it is used, the more accurate it becomes"; this effect is still a hypothesis to be verified.

**Our value proposition (one sentence)**:
> If you don't compete with it on "who has the highest score" - that will cost you a lot of money. We occupy the corners of its structure that are out of reach:
> **With very little cost (a small number of AI calls, no repeated trial and error), a solution is provided that is close, explainable, and more accurate the more it is used. **

For example: MLE-STAR is like a genius engineer who can "try violently a hundred times". The result is good, but it is expensive, slow, does not explain, and cannot remember the experience. Jiaozi is like a consultant who is knowledgeable, cheap, can explain things clearly, and understands you better the more you use it. For the vast majority of users who "don't want to spend money repeatedly adjusting parameters, but just want a solution that works", the latter is more practical.

> **Important Honesty**: We have also evaluated scoring techniques such as "model soup" and "integration", and concluded that they are public technologies and can be used by competing products.
> Moreover, they conflict with our cost advantage, so we do not regard them as selling points - this kind of "tried and judged not cost-effective" analysis itself is also the rigorous point of the project.

---

## 6. Current progress

**✅ Integrated (located at current `main`, can be demonstrated now)**:
- Complete module 1→2→3→4 pipeline (one sentence + data set → runnable code)
- Constraint-aware selection (budget filtering + cost model)
- Natural Language Recognition Evaluation Metrics
- Training improvements that allow large models (DINOv2) to "fairly compete"
- Hyperparameter recipe layer v0 (regular version, has been transferred from module 3 to module 4)
- A recommender that can accumulate and explain (memory + LogME + explanation)
- Cost metering (logging AI calls, training times, and wall clocks)
- Code-generated "If an error occurs, correct it first and then return to the template"
- notebook that can be run in the cloud (Google Colab) with one click
- 5 folds pairing results for CE vs focal on Cassava; the conclusions are limited to this data set

**🔬 To be verified/under planning**:
- Use real running records to verify whether the recommender can continuously improve ranking
- SIIM-ISIC Extremely unbalanced medical table CE vs focal Pairing experiment
- Real Kaggle benchmark and public scores; unfinished scores are not estimated

**To-do/next step** (see `docs/next_steps.md` for details):
- "Feed the memory of the recommender with real data", so that "the more you use it, the more accurate it will be" to truly turn into reality
- Complete the SIIM-ISIC pairing experiment and review whether it is necessary to update the knowledge base default policy
- Compare the "Quality vs Cost" with the competing product AIDE (**This picture is evidence of the core selling point**)
- Recipe layer v1 (let AI propose hyperparameters and rules serve as guardrails)
- Automatically expand the knowledge base (so that selection is not stuck in the small knowledge base maintained manually)
- ensemble (integrated) direction - ** to be determined **, because its benefits become smaller on the new generation of large models, need to be discussed

---

## 7. Summary for non-technical students

1. **What it solves**: Let people who don't understand AI get a set of usable image recognition solutions + codes with "one sentence + one data set".
2. **How ​​to do it**: Four-step assembly line - understand the requirements → look at the data → select the brain → produce the code.
3. **Why is it valuable**: Not "the most expensive and most accurate", but **cheap, explainable, budget-abiding, and more accurate the more you use it** - This is a corner that big competing products cannot achieve in their structure.
4. **Where are we now**: The core links are all connected and can run in the cloud; the next step is to use real data to verify "the more you use it, the more accurate it will be" and make a "quality vs cost" comparison chart.

> One sentence elevator statement:
> **"Jiaozi is the selection consultant for the image AI - he will give you a executable solution based on your requirements in one sentence, it is cheap, can be explained, and the more you use it, the better he will understand you;
> We do not compete with violent competitors that burn money, but occupy the corner of "high cost performance + explainability + accumulation" that they cannot reach. "**
