from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import openai
import os


app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
def calculate_adjustment(dB, target_dB):
    adjustment = target_dB - dB
    return round(max(-16, min(6, adjustment)), 0)

def generate_eq_settings(df, target_dB, num_bands=10, min_gap=15):
    df['Difference'] = (df['dB'] - target_dB).abs()
    sorted_df = df.sort_values(by='Difference', ascending=False)

    eq_settings = []
    selected_indices = []

    for _, row in sorted_df.iterrows():
        freq = row['Frequency']
        dB = row['dB']
        idx = row.name

        if all(abs(idx - selected_idx) >= min_gap for selected_idx in selected_indices):
            adjustment = calculate_adjustment(dB, target_dB)
            eq_settings.append((freq, adjustment))
            selected_indices.append(idx)

        if len(eq_settings) >= num_bands:
            break

    return eq_settings

def generate_eq_settings_kmeans(df, target_dB, num_bands=10):
    kmeans = KMeans(n_clusters=num_bands, n_init="auto")
    df["Cluster"] = kmeans.fit_predict(df[["Frequency"]])

    eq_settings = []
    for cluster in range(num_bands):
        cluster_data = df[df["Cluster"] == cluster]
        avg_freq = cluster_data["Frequency"].mean()
        avg_dB = cluster_data["dB"].mean()
        adjustment = calculate_adjustment(avg_dB, target_dB)
        eq_settings.append((avg_freq, adjustment))

    return eq_settings

def generate_eq_settings_logspaced(df, target_dB, num_bands=10):
    frequencies = np.logspace(np.log10(df["Frequency"].min()), np.log10(df["Frequency"].max()), num_bands)
    eq_settings = []

    for freq in frequencies:
        closest_idx = (df["Frequency"] - freq).abs().idxmin()
        closest_freq = df["Frequency"].iloc[closest_idx]
        closest_dB = df["dB"].iloc[closest_idx]
        adjustment = calculate_adjustment(closest_dB, target_dB)
        eq_settings.append((closest_freq, adjustment))

    return eq_settings
    
def ask_chatgpt_eq(freq_db_data, target_dB):
    prompt = (
        "Given this frequency response data:\n"
        + "\n".join([f"{freq} Hz: {dB} dB" for freq, dB in freq_db_data])
        + f"\n\nTarget dB: {target_dB}\n"
        "Generate 10 EQ band settings in the format: Frequency (Hz), Gain/Cut (dB),"
        " spaced out reasonably and only applying gain/cut within ±16 dB.\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a signal processing expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"ChatGPT Error: {e}"


def format_eq_settings(eq_settings, title):
    lines = [f"{title}", f"{'Frequency (Hz)':<15}{'Gain/Cut (dB)':<15}", "-" * 30]
    for freq, adjustment in eq_settings:
        lines.append(f"{freq:<15.2f}{adjustment:<15.0f}")
    return "\n".join(lines)

@app.route("/", methods=["GET", "POST"])
def index():
    output_text = ""
    input_data = ""

    if request.method == "POST":
        input_data = request.form.get("input_data", "").strip()
        if not input_data:
            output_text = "Error: Please enter input data."
        else:
            try:
                input_lines = [line for line in input_data.splitlines() if line.strip()]
                input_list = [tuple(map(float, line.split())) for line in input_lines]
                df = pd.DataFrame(input_list, columns=["Frequency", "dB"])

                target_freq = 600
                closest_idx = (df["Frequency"] - target_freq).abs().idxmin()
                target_dB = df.loc[closest_idx, "dB"]

                eq1 = generate_eq_settings(df, target_dB)
                eq2 = generate_eq_settings_kmeans(df, target_dB)
                eq3 = generate_eq_settings_logspaced(df, target_dB)
                chatgpt_eq = ask_chatgpt_eq_eq([(row["Frequency"], row["dB"]) for _, row in df.iterrows()], target_dB)

                output_text = "\n\n".join([
                    format_eq_settings(eq1, "EQ Settings - Highest to lowest"),
                    format_eq_settings(eq2, "EQ Settings - KMeans-Based"),
                    format_eq_settings(eq3, "EQ Settings - Log-Spaced Frequencies"),
                    "EQ Settings - ChatGPT-Based\n" + chatgpt_eq
                ])
            except Exception as e:
                output_text = f"Error processing data: {e}"

    html = """
    <!doctype html>
    <html>
    <head>
        <title>EQ Settings Generator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            textarea { width: 100%; height: 300px; font-family: monospace; font-size: 14px; }
            pre { background: #f0f0f0; padding: 15px; white-space: pre-wrap; }
            button { padding: 10px 20px; font-size: 16px; }
            label { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>EQ Settings Generator</h1>
        <form method="POST">
            <label for="input_data">Paste your Frequency and dB data here (two columns, space or tab separated):</label><br>
            <textarea id="input_data" name="input_data" placeholder="e.g. 20  -10\n50  -5\n100 0\n...">{{input_data}}</textarea><br><br>
            <button type="submit">Generate EQ Settings</button>
        </form>
        <h2>Generated EQ Settings:</h2>
        <pre>{{output_text}}</pre>
    </body>
    </html>
    """
    return render_template_string(html, output_text=output_text, input_data=input_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
