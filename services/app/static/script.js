const fileInput = document.getElementById('file');
const dropzone = document.getElementById('drop-text');
const preview = document.getElementById('preview');
const submit = document.getElementById('submit');
const reloadBtn = document.getElementById('reload');
const result = document.getElementById('result');
const reloadStatus = document.getElementById('reload-status');
const error = document.getElementById('error');

let selected = null;

function showError(msg) {
    error.textContent = msg;
}

function clearResult() {
    result.textContent = '';
    error.textContent = '';
}

fileInput.addEventListener('change', () => setImage(fileInput.files[0]));

dropzone.parentElement.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.parentElement.classList.add('dragover');
});

dropzone.parentElement.addEventListener('dragleave', () => {
    dropzone.parentElement.classList.remove('dragover');
});

dropzone.parentElement.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.parentElement.classList.remove('dragover');
    setImage(e.dataTransfer.files[0]);
});

function setImage(file) {
    clearResult();
    if (!file || !file.type.startsWith('image/')) {
        showError('Please choose an image file.');
        return;
    }
    selected = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    submit.disabled = false;
}

submit.addEventListener('click', async () => {
    if (!selected) return;
    clearResult();
    submit.disabled = true;
    submit.textContent = 'Classifying...';

    const form = new FormData();
    form.append('image', selected);

    try {
        const resp = await fetch('/classify', { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const name = data.label === 1 ? 'Dog' : 'Cat';
        result.textContent = `${name} (${((data.prob * 100) % 50 + 50).toFixed(1)}%)`;
    } catch (err) {
        showError('Classification failed: ' + err.message);
    } finally {
        submit.disabled = false;
        submit.textContent = 'Classify';
    }
});

reloadBtn.addEventListener('click', async () => {
    clearResult();
    reloadStatus.hidden = false;
    reloadStatus.className = 'status';
    reloadStatus.textContent = 'Reloading...';
    reloadBtn.disabled = true;

    try {
        const resp = await fetch('/reload-model', { method: 'POST' });
        if (!resp.ok) {
            const detail = await resp.json().catch(() => null);
            throw new Error(detail?.detail ?? `HTTP ${resp.status}`);
        }
        reloadStatus.textContent = 'Model reloaded.';
    } catch (err) {
        reloadStatus.className = 'error';
        reloadStatus.textContent = 'Reload failed: ' + err.message;
    } finally {
        reloadBtn.disabled = false;
    }
});
