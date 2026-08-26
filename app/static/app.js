const form = document.querySelector('#question-form');
const input = document.querySelector('#question');
const askButton = document.querySelector('#ask-button');

const result = document.querySelector('#result');
const answer = document.querySelector('#answer');
const citations = document.querySelector('#citations');
const badge = document.querySelector('#mode-badge');
const latency = document.querySelector('#latency');
const fallback = document.querySelector('#fallback');

const uploadForm = document.querySelector('#upload-form');
const pdfFile = document.querySelector('#pdf-file');
const uploadKey = document.querySelector('#upload-key');
const uploadButton = document.querySelector('#upload-button');
const uploadStatus = document.querySelector('#upload-status');
const filePickerText = document.querySelector('#file-picker-text');
const selectedFile = document.querySelector('#selected-file');


/* =========================
   EXAMPLE QUESTIONS
========================= */

document.querySelectorAll('[data-question]').forEach((button) => {
    button.addEventListener('click', () => {
        input.value = button.dataset.question;
        input.focus();
    });
});


/* =========================
   ASK THE ARCHIVE
========================= */

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    if (!input.value.trim()) {
        return;
    }

    askButton.disabled = true;
    askButton.querySelector('span').textContent = 'SEARCHING ARCHIVE…';

    const controller = new AbortController();

    const timeout = setTimeout(() => {
        controller.abort();
    }, 60000);

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: input.value.trim()
            }),
            signal: controller.signal
        });

        const text = await response.text();

        let data;

        try {
            data = JSON.parse(text);
        } catch {
            throw new Error(
                `Server returned an invalid response (${response.status}).`
            );
        }

        if (!response.ok) {
            throw new Error(
                data.error || 'The archive could not answer that request.'
            );
        }

        result.classList.remove('hidden');

        const isFallback = data.mode === 'retrieval_fallback';

        badge.textContent = isFallback
            ? 'SOURCE-ONLY FALLBACK'
            : 'GROUNDED ANSWER';

        badge.classList.toggle('fallback', isFallback);

if (data.mode === 'subject_required') {
    answer.textContent =
        data.error ||
        'Please specify a subject: CNS, BDA, PC, or IOT.';

    badge.textContent = 'SUBJECT REQUIRED';
    badge.classList.remove('fallback');

    fallback.classList.add('hidden');
    citations.replaceChildren();

} else {
    answer.innerHTML =
        data.answer
            ? marked.parse(data.answer)
            : 'The language model is unavailable, so here is the retrieved evidence directly.';
}
        fallback.classList.toggle('hidden', !isFallback);

        if (isFallback) {
            fallback.textContent =
                `Generation unavailable: ${data.fallback_reason || 'unknown reason'}. ` +
                'The passages below are still usable source evidence.';
        } else {
            fallback.textContent = '';
        }

        if (data.latency_ms && data.cache) {
            latency.textContent =
                `${data.latency_ms.total} ms TOTAL · ` +
                `${data.cache.hit ? 'CACHE HIT' : 'FRESH RETRIEVAL'}`;
        }

        citations.replaceChildren();

        if (Array.isArray(data.retrieved_chunks)) {
            data.retrieved_chunks.forEach((item, index) => {
                const card = document.createElement('article');
                card.className = 'citation';

                const source = document.createElement('strong');

                source.textContent =
                    `[${index + 1}] ` +
                    `${item.citation.document} · p. ${item.citation.page}`;

                const excerpt = document.createElement('p');
                excerpt.textContent = item.text;

                card.appendChild(source);
                card.appendChild(excerpt);

                citations.appendChild(card);
            });
        }

        result.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });

    } catch (error) {

        result.classList.remove('hidden');

        badge.textContent = 'REQUEST FAILED';
        badge.classList.add('fallback');

        if (error.name === 'AbortError') {
            answer.textContent =
                'The archive took too long to respond. Check the Render logs.';
        } else {
            answer.textContent = error.message;
        }

        fallback.classList.add('hidden');
        citations.replaceChildren();

    } finally {
        clearTimeout(timeout);

        askButton.disabled = false;
        askButton.querySelector('span').textContent = 'ASK THE ARCHIVE';
    }
});


/* =========================
   FILE SELECTION
========================= */

if (pdfFile) {
    pdfFile.addEventListener('change', () => {

        if (!pdfFile.files.length) {
            filePickerText.textContent = 'CHOOSE PDF OR TXT';
            selectedFile.textContent = 'No file selected';
            return;
        }

        const file = pdfFile.files[0];

        filePickerText.textContent = 'CHANGE FILE';

        selectedFile.textContent =
            `${file.name} · ${formatFileSize(file.size)}`;
    });
}


function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


/* =========================
   UPLOAD
========================= */

if (uploadForm) {

    uploadForm.addEventListener('submit', async (event) => {

        event.preventDefault();

        if (!pdfFile.files.length) {
            showUploadStatus(
                'Please choose a PDF or TXT file first.',
                false
            );
            return;
        }

        const key = uploadKey.value.trim();

        if (!key) {
            showUploadStatus(
                'Enter the upload key before uploading.',
                false
            );

            uploadKey.focus();
            return;
        }

        uploadButton.disabled = true;
        uploadButton.querySelector('span').textContent = 'INDEXING…';

        showUploadStatus(
            'Uploading and indexing document…',
            true
        );

        try {

            const formData = new FormData();

            formData.append(
                'file',
                pdfFile.files[0]
            );

            const response = await fetch('/upload', {
                method: 'POST',
                headers: {
                    'X-Upload-Key': key
                },
                body: formData
            });

            const text = await response.text();

            let data;

            try {
                data = JSON.parse(text);
            } catch {
                throw new Error(
                    `Upload failed (${response.status}).`
                );
            }

            if (!response.ok) {

                if (response.status === 403) {
                    throw new Error(
                        'Upload unauthorized. Check the upload key.'
                    );
                }

                throw new Error(
                    data.error || 'Upload failed.'
                );
            }

            const chunks =
                data.stats &&
                data.stats.chunks !== undefined
                    ? data.stats.chunks
                    : '?';

            showUploadStatus(
                `✓ ${data.message} (${chunks} total chunks ready)`,
                true
            );

            pdfFile.value = '';

            filePickerText.textContent = 'CHOOSE PDF OR TXT';
            selectedFile.textContent = 'No file selected';

        } catch (error) {

            showUploadStatus(
                `✕ ${error.message}`,
                false
            );

        } finally {

            uploadButton.disabled = false;

            uploadButton.querySelector('span').textContent =
                'UPLOAD & INDEX';
        }
    });
}


function showUploadStatus(message, success) {

    uploadStatus.classList.remove('hidden');

    uploadStatus.textContent = message;

    uploadStatus.style.color =
        success ? '#4ade80' : '#ef4444';
}