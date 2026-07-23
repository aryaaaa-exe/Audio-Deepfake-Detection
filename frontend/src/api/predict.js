import axios from "axios";

const API = "http://127.0.0.1:5000/predict";

export async function predictAudio(file) {

    const formData = new FormData();

    formData.append("audio", file);

    const response = await axios.post(
        API,
        formData,
        {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        }
    );

    return response.data;
}