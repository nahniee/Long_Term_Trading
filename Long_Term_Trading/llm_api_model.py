import openai

class BaseAIModel:
    def __init__(self):
        self.model = None

    def query_model(self):
        pass

class OpenAIModel(BaseAIModel):
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        openai.api_key = api_key  # Set the API key
        self.model = openai
        self.model_name= model_name

    def query_model(self, prompt):
        resp_content = None

        if self.model_name == "gpt-4o-mini-search-preview" or \
            self.model_name == "gpt-4o-search-preview":

            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            resp_content = response.choices[0].message.content  
        else:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            resp_content = response.choices[0].message.content
            
        return resp_content

