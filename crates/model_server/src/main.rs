use askama_axum::Template;
use axum::routing::get;
use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use axum_auth::AuthBearer;
use color_eyre::eyre::{bail, Result};
use linkify::LinkFinder;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::{fs::OpenOptions, net::SocketAddr};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
use tracing::{error, info};
use voca_rs::strip;

static GRAPH: OnceCell<Graph> = OnceCell::new();
static MODEL: OnceCell<SavedModelBundle> = OnceCell::new();

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    // initialize tracing
    tracing_subscriber::fmt::init();
    info!("Starting up");

    let model_path = match std::env::var("MODEL_PATH") {
        Ok(val) => val,
        Err(_) => bail!("Missing MODEL_PATH env var"),
    };

    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_path)?;
    GRAPH.set(graph).unwrap();
    MODEL.set(bundle).unwrap();

    // build our application with a route
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        // `GET /test` goes to `test`
        .route("/test", post(test))
        // `POST /submit` goes to `submit`
        .route("/submit", post(submit))
        .route("/submit_review", post(submit_for_review));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

async fn index() -> IndexTemplate {
    IndexTemplate {}
}

async fn test(Json(payload): Json<TestData>) -> impl IntoResponse {
    let bundle = MODEL.get().unwrap();
    let graph = GRAPH.get().unwrap();
    let session = &bundle.session;
    let meta = bundle.meta_graph_def();
    let signature = meta
        .get_signature(tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        .unwrap();
    let input_info = signature.get_input("input_1").unwrap();
    let output_info = signature.get_output("output_1").unwrap();

    let input_op = graph
        .operation_by_name_required(&input_info.name().name)
        .unwrap();
    let output_op = graph
        .operation_by_name_required(&output_info.name().name)
        .unwrap();

    let tensor: Tensor<String> = Tensor::from(&[payload.input_data.clone()]);
    let mut args = SessionRunArgs::new();
    args.add_feed(&input_op, 0, &tensor);

    let out = args.request_fetch(&output_op, 0);

    session
        .run(&mut args)
        .expect("Error occurred during calculations");
    let out_res: f32 = args.fetch(out).unwrap()[0];

    let response = Prediction {
        input_data: payload.input_data,
        score: out_res,
    };

    (StatusCode::OK, Json(response))
}

async fn submit(
    Json(payload): Json<SubmitData>,
    AuthBearer(token): AuthBearer,
) -> impl IntoResponse {
    let access_token = match std::env::var("ACCESS_TOKEN") {
        Ok(val) => val,
        Err(_) => {
            error!("Missing ACCESS_TOKEN env var");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };
    if token != access_token {
        return StatusCode::UNAUTHORIZED;
    }

    // TODO implement
    StatusCode::NOT_IMPLEMENTED
}

async fn submit_for_review(
    Json(payload): Json<SubmitReview>,
    AuthBearer(token): AuthBearer,
) -> impl IntoResponse {
    let access_token = match std::env::var("ACCESS_TOKEN") {
        Ok(val) => val,
        Err(_) => {
            error!("Missing ACCESS_TOKEN env var");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };
    if token != access_token {
        return StatusCode::UNAUTHORIZED;
    }

    std::fs::create_dir_all("./data/").unwrap();
    let file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open("./data/review.txt");

    // Sanitize
    // We remove newlines, html tags and links
    let sanitized = strip::strip_tags(&payload.input_data);
    let sanitized = sanitized.replace(['\r', '\n'], " ");
    let mut sanitized = trim_whitespace(&sanitized);
    let mut finder = LinkFinder::new();
    let cloned_sanitized = sanitized.clone();
    finder.url_must_have_scheme(false);
    let links: Vec<_> = finder.links(&cloned_sanitized).collect();
    for link in links {
        sanitized = sanitized.replace(link.as_str(), " ");
    }
    match file {
        Ok(mut file) => {
            if let Err(e) = writeln!(file, "{}", sanitized) {
                eprintln!("Couldn't write to file: {}", e);
                return StatusCode::INTERNAL_SERVER_ERROR;
            }
        }
        Err(e) => {
            eprintln!("Couldn't open file: {}", e);
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    }

    StatusCode::OK
}

fn trim_whitespace(s: &str) -> String {
    let mut new_str = s.trim().to_owned();
    let mut prev = ' '; // The initial value doesn't really matter
    new_str.retain(|ch| {
        let result = ch != ' ' || prev != ' ';
        prev = ch;
        result
    });
    new_str
}

#[derive(Deserialize, Serialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
struct TestData {
    input_data: String,
}

#[derive(Deserialize, Serialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
struct Prediction {
    input_data: String,
    score: f32,
}

#[derive(Deserialize, Serialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
struct SubmitData {
    input_data: String,
    spam: bool,
}

#[derive(Deserialize, Serialize)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
struct SubmitReview {
    input_data: String,
}

#[cfg(test)]
mod test {
    use crate::{Prediction, SubmitData, SubmitReview, TestData};

    #[test]
    fn generate_schema() {
        let test_data_schema = schemars::schema_for!(TestData);
        let prediction_schema = schemars::schema_for!(Prediction);
        let submit_data_schema = schemars::schema_for!(SubmitData);
        let submit_review_schema = schemars::schema_for!(SubmitReview);

        std::fs::create_dir_all("./schemas").unwrap();
        std::fs::write(
            "./schemas/test_data.json",
            serde_json::to_string_pretty(&test_data_schema).unwrap(),
        )
        .unwrap();
        std::fs::write(
            "./schemas/prediction.json",
            serde_json::to_string_pretty(&prediction_schema).unwrap(),
        )
        .unwrap();
        std::fs::write(
            "./schemas/submit_data.json",
            serde_json::to_string_pretty(&submit_data_schema).unwrap(),
        )
        .unwrap();
        std::fs::write(
            "./schemas/submit_review.json",
            serde_json::to_string_pretty(&submit_review_schema).unwrap(),
        )
        .unwrap();
    }
}
