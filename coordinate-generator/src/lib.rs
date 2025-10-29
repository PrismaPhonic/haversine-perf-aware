use rand::{
    distr::{Distribution, Uniform},
    rngs::ThreadRng,
};
use serde::{Deserialize, Serialize};
use serde_json::ser::Formatter;
use std::{any::Any, fs::File, io::BufWriter, path::PathBuf};

use clap::Parser;

pub const EARTH_RADIUS: f64 = 6372.8;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long)]
    pairs: u32,

    #[arg(short, long, default_value = "coordinates.json")]
    output_file: PathBuf,

    #[arg(short, long, default_value = "answers.json")]
    answers_file: PathBuf,
}

#[derive(Serialize, Deserialize)]
pub struct CoordinatePairs {
    pub pairs: Vec<CoordinatePair>,
}

#[derive(Serialize, Deserialize)]
pub struct CoordinatePair {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

#[derive(Serialize, Deserialize)]
pub struct HaversineAnswers {
    pairs: Vec<f64>,
}

#[inline(always)]
fn generate_pair(rng: &mut ThreadRng, x: &Uniform<f64>, y: &Uniform<f64>) -> CoordinatePair {
    CoordinatePair {
        x0: x.sample(rng),
        y0: y.sample(rng),
        x1: x.sample(rng),
        y1: y.sample(rng),
    }
}

#[inline(always)]
pub fn generate_pairs() {
    let Args {
        pairs,
        output_file,
        answers_file,
    } = Args::parse();

    let mut rng = rand::rng();
    let x_gen =
        Uniform::new_inclusive(-180.0, 180.0).expect("Failed to create uniform distribution");
    let y_gen = Uniform::new_inclusive(-90.0, 90.0).expect("Failed to create uniform distribution");

    let sum_coef = 1.0 / pairs as f64;
    let mut pairs_out = Vec::with_capacity(pairs as usize);
    let mut answers_out = Vec::with_capacity(pairs as usize);
    let mut total = 0.0;
    for _ in 0..pairs {
        let pair = generate_pair(&mut rng, &x_gen, &y_gen);

        let distance = reference_haversine(pair.x0, pair.y0, pair.x1, pair.y1);
        total += sum_coef * distance;

        answers_out.push(distance);
        pairs_out.push(pair);
    }

    println!("Expected sum: {}", total);

    let coordinate_pairs = CoordinatePairs { pairs: pairs_out };

    let file = File::create(output_file).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    let mut ser = serde_json::Serializer::with_formatter(writer, CustomLengthFloatFormatter {});
    coordinate_pairs
        .serialize(&mut ser)
        .expect("Failed to write coordinates to disk");

    let answers = HaversineAnswers { pairs: answers_out };
    let file = File::create(answers_file).expect("Failed to create answers file");
    let writer = BufWriter::new(file);
    let mut ser = serde_json::Serializer::with_formatter(writer, CustomLengthFloatFormatter {});
    answers
        .serialize(&mut ser)
        .expect("Failed to write answers to disk");
}

pub struct CustomLengthFloatFormatter;

impl Formatter for CustomLengthFloatFormatter {
    #[inline]
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if value <= -100.0 {
            writer.write_fmt(format_args!("{:.12}", value))
        } else if value <= -10.0 {
            writer.write_fmt(format_args!("{:.13}", value))
        } else if value.is_negative() {
            writer.write_fmt(format_args!("{:.14}", value))
        } else if value < 10.0 {
            writer.write_fmt(format_args!("{:.15}", value))
        } else if value < 100.0 {
            writer.write_fmt(format_args!("{:.14}", value))
        } else {
            writer.write_fmt(format_args!("{:.13}", value))
        }
    }
}

#[inline(always)]
fn reference_haversine(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let mut lat1 = y0;
    let mut lat2 = y1;
    let lon1 = x0;
    let lon2 = x1;

    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    lat1 = lat1.to_radians();
    lat2 = lat2.to_radians();

    let a = ((d_lat / 2.0).sin()).powf(2.0)
        + (lat1.cos() * lat2.cos() * ((d_lon / 2.0).sin()).powf(2.0));
    let c = 2.0 * (a.sqrt()).asin();

    EARTH_RADIUS * c
}
