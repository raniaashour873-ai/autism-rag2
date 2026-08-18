const API_BASE_URL = "http://127.0.0.1:8000";


/**
 * Send a clinical question to the FastAPI RAG backend.
 *
 * @param {string} question
 * @param {number} topK
 * @returns {Promise<object>}
 */
export async function askClinicalQuestion(
    question,
    topK = 5
) {

    const response = await fetch(
        `${API_BASE_URL}/ask`, {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
            },

            body: JSON.stringify({
                question: question,
                top_k: topK,
            }),
        }
    );


    // =========================================================
    // HANDLE API ERRORS
    // =========================================================

    if (!response.ok) {

        let message =
            `API request failed (${response.status})`;

        try {

            const errorData =
                await response.json();

            /*
             * FastAPI usually returns:
             *
             * {
             *   "detail": "..."
             * }
             *
             * IMPORTANT:
             * No space between ? and .
             */

            if (errorData.detail) {

                message =
                    errorData.detail;

            }

        } catch (parseError) {

            console.warn(
                "Could not parse API error response:",
                parseError
            );

        }

        throw new Error(message);
    }


    // =========================================================
    // PARSE SUCCESS RESPONSE
    // =========================================================

    const data =
        await response.json();


    return data;
}


/**
 * Check whether the FastAPI backend is alive.
 *
 * @returns {Promise<object>}
 */
export async function checkApiHealth() {

    const response =
        await fetch(
            `${API_BASE_URL}/`
        );


    if (!response.ok) {

        throw new Error(
            "Backend health check failed."
        );

    }


    return response.json();
}