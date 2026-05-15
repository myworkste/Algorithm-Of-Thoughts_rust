use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct LlmClient {
    http_client: Client,
    server_url: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    prompt: String,
    n: usize,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Deserialize)]
struct GenerateResponse {
    choices: Vec<String>,
}

impl LlmClient {
    pub fn new(server_url: String) -> Self {
        Self {
            http_client: Client::new(),
            server_url,
        }
    }

    pub async fn generate_text(&self, prompt: &str, k: usize) -> Result<Vec<String>, String> {
        let req = GenerateRequest {
            prompt: prompt.to_string(),
            n: k,
            max_tokens: 400,
            temperature: 0.5,
        };

        // Note: No API key is sent or handled here as per the security architecture.
        // We route all LLM generation requests to our external cloud server API.
        let res = self
            .http_client
            .post(&self.server_url)
            .json(&req)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let resp_data: GenerateResponse = res.json().await.map_err(|e| e.to_string())?;
        Ok(resp_data.choices)
    }

    pub async fn generate_thoughts(
        &self,
        state: &str,
        k: usize,
        initial_prompt: &str,
    ) -> Result<Vec<String>, String> {
        let prompt = format!(
            "Accomplish the task below by decomposing it as many very explicit subtasks as possible, be very explicit and thorough denoted by a search process...\n\n########## OBJECTIVE\n{}\n###################\n\nState:\n{}",
            initial_prompt, state
        );
        self.generate_text(&prompt, k).await
    }

    pub async fn generate_solution(
        &self,
        initial_prompt: &str,
        state: &str,
    ) -> Result<String, String> {
        let prompt = format!(
            "Generate a series of solutions to comply with the user's instructions...\n\n###'{}'\n\n###\nDevise the best possible solution for the task: {}",
            state, initial_prompt
        );
        let answers = self.generate_text(&prompt, 1).await?;
        Ok(answers.into_iter().next().unwrap_or_default())
    }

    pub async fn evaluate_states(
        &self,
        states: &[String],
        initial_prompt: &str,
    ) -> Result<Vec<f64>, String> {
        let mut results = Vec::new();
        for state in states {
            let prompt = format!(
                "To achieve the following goal: '{}', pessimistically value the context of the past solutions...\nPast solutions:\n\n{}\nEvaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE",
                initial_prompt, state
            );
            let response = self.generate_text(&prompt, 1).await?;
            let value_str = response.into_iter().next().unwrap_or_default();
            let value: f64 = value_str.trim().parse().unwrap_or(0.0);
            results.push(value);
        }
        Ok(results)
    }
}
