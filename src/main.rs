use aot_rs::{AoT, LlmClient};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let client = Arc::new(LlmClient::new(
        "https://api.internal.cloud/generate".to_string(),
    ));
    let aot = AoT::new(
        2,
        10,
        1.0,
        0.5,
        0.4,
        "Calculate 24 using numbers 14 8 8 2".to_string(),
        client,
    );

    println!("Starting solve...");
    if let Some(solution) = aot.solve().await {
        println!("Solution: {}", solution);
    } else {
        println!("No solution found.");
    }
}
