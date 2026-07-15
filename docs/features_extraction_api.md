# 🥟 Dumpling Project (Jiaozi) - Natural Language Feature Extraction Module API Documentation

**Responsible person**: Zhang Mingda **Version**: v1.2 **Update date**: 2026-04-03

---


## Module overview
This module is responsible for processing natural language input from users (e.g., laboratory researchers). By calling the Qwen large language model (qwen-plus), the user's core intentions and model features are understood and extracted, and the final output is a structured string list, which provides accurate conditions for the downstream RAG Agent to Huggingface to retrieve the model.

---

## Interface call

- **Interface path**: `/api/v1/features_extraction`
- **Function Description**: Receives the user's natural language text and returns the parsed model feature list.
- **Content Type**: `list`

### 1. Request parameters (Request Body)

| Field name | Required | type | illustrate |
| :--- | :---: | :--- | :--- |
| `user_message` | yes | `string` | User original natural language input |

**Request Example:**
```json
{
  "user_message": "I need a medical image segmentation model for MRI MRI images, preferably implemented by PyTorch, which outputs the Mask image and has a high accuracy index."
}
```
### 2. Response parameters (Response Body)

| Field name              | Required | type | illustrate |
|:-----------------| :---: | :--- | :--- |
| `system_message` | yes | `string` | Extracted structured feature data |

**Response example:**
```json
{
  "system_message":["Domain: Medical Image", "Task: image segmentation", "Accuracy: accuracy", "Accuracy_range: High", "is_local_train: null", "Graphics_card: null", "Input: pictures", "Output: pictures", "Size: null", "Library / Framework: PyTorch", "Input_Language: Chinese", "Output_Language: Chinese", "License: null"]
  }
}
```
---

## Large model prompt
```
[Identity] Huggingface model retrieval expert.
[Task] Extract search features from natural language.
[Format] list of pure string, in which the elements contain the following 14 dimensions in order. key (dimension items) in each element of list are all in English. The value is consistent with the user's original input language (for example, if the user input language is English, the value will capture the original English text). For the style, please refer to: When the input language is Chinese ["Domain: Biology", "Task: text generates text"], when the input language is English ["Domain: Biology", "Task: text to text"];
[Dimension] must be reviewed and extracted:
1.Field(Domain)
2.Task type (Task)
3. Model accuracy evaluation parameters (Accuracy)
4. Model accuracy evaluation parameter range (Accuracy_range)
5. Whether to train locally (is_local_train)
6. Graphics card model (Graphics_card)
7. Whether to train locally
8. Enter (Input)
9. Output (Output)
10. Parameter magnitude (Size)
11. Frame (Library / Framework)
12.Input language (Input_Language)
13.Output language (Output_Language)
14. Agreement (License)
**Note:
1. The output content of the two dimensions of input/output is only ["text", "picture", "audio", "video"] (the output content is consistent with the user's actual input language. For example, if the user input is English, the output of the two dimensions of input/output is ["Text", "Image", "Audio", "Video"]);
2. If the user proposes a specific value for any of the above dimensions, the value will be captured as the output. If no specific value is proposed, null will be used as the output. If the user does not fully mention the above dimensions, all dimensions will still be completed in list, and the unmentioned dimensions will be uniformly used as null as the output;
3. If a specific output language is mentioned, only the specific output language will be used. If no output language is mentioned, the value of [Output Language] will be set to "English" by default;
[Rule] Only extract dimensions that are mentioned by the user or can be reasonably inferred, and dimensions that are not mentioned are simply ignored. (No greetings allowed, no markdown symbols allowed).
```

*Note: User input is divided into [lite version] and [customized version]. [lite version] only includes the following dimensions: ["field (Domain)", "task type (Task)", "whether to train locally (is_local_train)", "graphics card model (Graphics_card)", "Input (Input)", "Output (Output)", "Input language (Input_Language)", "Output language (Output_Language)"], [customized version] includes all dimensions. This classification must be implemented interactively on the front end. This module only focuses on feature recognition;*

---

## User natural language example

- **case_1 (standard)**: `I need a medical image-segmentation model for MRI scans, preferably implemented in PyTorch. It should output mask images, prioritize high accuracy, and use Chinese as the language.`

Output: ["Domain: medical image", "Task: image segmentation", "Accuracy: accuracy", "Accuracy_range: high", "is_local_train: null", "Graphics_card: null", "Input: picture", "Output: picture", "Size: null", "Library / Framework: PyTorch", "Input_Language: Chinese", "Output_Language: Chinese", "License: null"]

- **case_2 (standard)**: `Is there a JAX implementation for satellite imagery segmentation or classification? Specifically, we are looking for a model that processes multispectral data to generate LULC (Land Use/Land Cover) maps, provided it allows for commercial use.`

Output: ["Domain: Remote Sensing", "Task: image to image", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: Image", "Output: Image", "Size: null", "Library / Framework: JAX", "Input_Language: English", "Output_Language: English", "License: commercial use"]

- **case_3 (vague)**: `Find me a model for an intelligent customer-service assistant. Any framework is fine.`

Output: ["Domain: Customer Service", "Task: Text Generate Text", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: text", "Output: text", "Size: null", "Library / Framework: null", "Input_Language: null", "Output_Language: English", "License: null"]

- **case_4 (industry jargon)**: `Our lab is working on a Chinese NER task. Recommend several state-of-the-art BERT variants that support multilingual output.`

Output: ["Domain: Natural Language Processing", "Task: Named Entity Recognition", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: text", "Output: text", "Size: null", "Library / Framework: null", "Input_Language: Chinese", "Output_Language: Multilingual", "License: null"]

- **case_5 (colloquial)**: `I need a model for Chinese sentiment analysis. Do not recommend an LLM with tens of billions of parameters because I cannot run it; suggest something lightweight with reasonable accuracy.`

Output: ["Domain: sentiment analysis", "Task: text generation text", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: text", "Output: text", "Size: lightweight", "Library / Framework: null", "Input_Language: Chinese", "Output_Language: English", "License: null"]

- **case_6 (colloquial)**: `Find a baseline model for forecasting financial data, primarily for time-series analysis.`

Output: ["Domain: Finance", "Task: Time Series Forecast", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: text", "Output: text", "Size: null", "Library / Framework: null", "Input_Language: Chinese", "Output_Language: English", "License: null"]




**2026/3/28 To-do**:
- Front-end model parameters and ranges can be selected by users and provide default values.
- Prepare two sets of user input, a simplified version and a customized version
- Information about joining [Card]
- The string List is reserved for English only
