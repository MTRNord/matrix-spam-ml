use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use axum_auth::AuthBearer;
use color_eyre::eyre::{bail, Result};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::{fs::OpenOptions, net::SocketAddr};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
use tracing::{debug, error, info};

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
        // `GET /test` goes to `test`
        .route("/test", post(test))
        // `POST /submit` goes to `submit`
        .route("/submit", post(submit))
        .route("/submit_review", post(submit_for_review));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}

async fn test(Json(payload): Json<TestData>) -> impl IntoResponse {
    let bundle = MODEL.get().unwrap();
    let graph = GRAPH.get().unwrap();
    let session = &bundle.session;
    let meta = bundle.meta_graph_def();
    debug!("Signatures: {:#?}", meta.signatures());
    let signature = meta
        .get_signature(tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        .unwrap();
    debug!("Inputs: {:#?}", signature.inputs());
    debug!("Outputs: {:#?}", signature.outputs());
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
    match file {
        Ok(mut file) => {
            if let Err(e) = writeln!(file, "{}", payload.input_data) {
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

#[derive(Deserialize, Serialize)]
struct TestData {
    input_data: String,
}

#[derive(Deserialize, Serialize)]
struct Prediction {
    input_data: String,
    score: f32,
}

#[derive(Deserialize, Serialize)]
struct SubmitData {
    input_data: String,
    spam: bool,
}

#[derive(Deserialize, Serialize)]
struct SubmitReview {
    input_data: String,
}
