use crate::client::LlmClient;
use async_recursion::async_recursion;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AoT {
    num_thoughts: usize,
    max_steps: usize,
    value_threshold: f64,
    pruning_threshold: f64,
    backtracking_threshold: f64,
    initial_prompt: String,
    client: Arc<LlmClient>,
    thought_cache_accepted: Arc<Mutex<HashMap<String, f64>>>,
    thought_cache_pruned: Arc<Mutex<HashMap<String, f64>>>,
    output: Arc<Mutex<Vec<(String, f64)>>>,
}

impl AoT {
    pub fn new(
        num_thoughts: usize,
        max_steps: usize,
        value_threshold: f64,
        pruning_threshold: f64,
        backtracking_threshold: f64,
        initial_prompt: String,
        client: Arc<LlmClient>,
    ) -> Self {
        Self {
            num_thoughts,
            max_steps,
            value_threshold,
            pruning_threshold,
            backtracking_threshold,
            initial_prompt,
            client,
            thought_cache_accepted: Arc::new(Mutex::new(HashMap::new())),
            thought_cache_pruned: Arc::new(Mutex::new(HashMap::new())),
            output: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[async_recursion]
    pub async fn dfs(&self, state: String, step: usize) {
        if step > self.max_steps {
            let value = {
                let accepted = self.thought_cache_accepted.lock().await;
                let pruned = self.thought_cache_pruned.lock().await;
                if let Some(&v) = accepted.get(&state) {
                    Some(v)
                } else if pruned.contains_key(&state) {
                    return;
                } else {
                    None
                }
            };

            let value = match value {
                Some(v) => v,
                None => {
                    let (_, v) = self.evaluate_thought(state.clone()).await;
                    self.thought_cache_accepted
                        .lock()
                        .await
                        .insert(state.clone(), v);
                    v
                }
            };

            self.output.lock().await.push((state.clone(), value));
            return;
        }

        let thoughts = {
            let accepted = self.thought_cache_accepted.lock().await;
            let pruned = self.thought_cache_pruned.lock().await;
            if accepted.contains_key(&state) {
                Some(vec![state.clone()])
            } else if pruned.contains_key(&state) {
                return;
            } else {
                None
            }
        };

        let thoughts = match thoughts {
            Some(t) => t,
            None => self.generate_and_filter_thoughts(state.clone()).await,
        };

        for next_state in thoughts {
            let state_value = {
                let accepted = self.thought_cache_accepted.lock().await;
                let pruned = self.thought_cache_pruned.lock().await;
                accepted
                    .get(&next_state)
                    .copied()
                    .unwrap_or_else(|| pruned.get(&next_state).copied().unwrap_or(0.0))
            };

            if state_value <= self.value_threshold {
                self.thought_cache_pruned
                    .lock()
                    .await
                    .insert(next_state.clone(), state_value);
                continue;
            }

            let child = format!("{}\n{}", state, next_state);
            self.dfs(child, step + 1).await;

            let best_value = {
                let out = self.output.lock().await;
                out.iter()
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max)
            };

            if best_value < self.backtracking_threshold {
                self.output.lock().await.pop();
                continue;
            }
        }
    }

    pub async fn generate_and_filter_thoughts(&self, state: String) -> Vec<String> {
        let thoughts = self
            .client
            .generate_thoughts(&state, self.num_thoughts, &self.initial_prompt)
            .await
            .unwrap_or_default();
        let evaluated_values = self
            .client
            .evaluate_states(&thoughts, &self.initial_prompt)
            .await
            .unwrap_or_default();

        let mut filtered = Vec::new();
        let mut accepted_lock = self.thought_cache_accepted.lock().await;
        let mut pruned_lock = self.thought_cache_pruned.lock().await;

        for (thought, &val) in thoughts.iter().zip(evaluated_values.iter()) {
            if val >= self.pruning_threshold {
                filtered.push(thought.clone());
                accepted_lock.insert(thought.clone(), val);
            } else {
                pruned_lock.insert(thought.clone(), val);
            }
        }

        filtered
    }

    pub async fn evaluate_thought(&self, state: String) -> (String, f64) {
        let thoughts = self
            .client
            .generate_thoughts(&state, 1, &self.initial_prompt)
            .await
            .unwrap_or_default();
        let thought = thoughts.into_iter().next().unwrap_or_default();
        let vals = self
            .client
            .evaluate_states(&[state.clone()], &self.initial_prompt)
            .await
            .unwrap_or_default();
        let val = vals.into_iter().next().unwrap_or(0.0);

        if val >= self.pruning_threshold {
            self.thought_cache_accepted
                .lock()
                .await
                .insert(state.clone(), val);
        } else {
            self.thought_cache_pruned
                .lock()
                .await
                .insert(state.clone(), val);
        }
        (thought, val)
    }

    pub async fn solve(&self) -> Option<String> {
        self.dfs(self.initial_prompt.clone(), 1).await;

        let best_state = {
            let out = self.output.lock().await;
            if out.is_empty() {
                return None;
            }
            out.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|x| x.0.clone())
        };

        if let Some(state) = best_state {
            if let Ok(solution) = self
                .client
                .generate_solution(&self.initial_prompt, &state)
                .await
            {
                Some(solution)
            } else {
                Some(state)
            }
        } else {
            None
        }
    }
}
