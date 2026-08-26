const form = document.querySelector('#question-form');
const input = document.querySelector('#question');
const askButton = document.querySelector('#ask-button');
const result = document.querySelector('#result');
const answer = document.querySelector('#answer');
const citations = document.querySelector('#citations');
const badge = document.querySelector('#mode-badge');
const latency = document.querySelector('#latency');
const fallback = document.querySelector('#fallback');

document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => {
  input.value = button.dataset.question;
  input.focus();
}));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  askButton.disabled = true;
  askButton.querySelector('span').textContent = 'SEARCHING ARCHIVE…';
  try {
    const response = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:input.value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The archive could not answer that request.');
    result.classList.remove('hidden');
    const isFallback = data.mode === 'retrieval_fallback';
    badge.textContent = isFallback ? 'SOURCE-ONLY FALLBACK' : 'GROUNDED ANSWER';
    badge.classList.toggle('fallback', isFallback);
    answer.textContent = data.answer || 'The language model is unavailable, so here is the retrieved evidence directly.';
    fallback.classList.toggle('hidden', !isFallback);
    fallback.textContent = isFallback ? `Generation unavailable: ${data.fallback_reason}. The passages below are still usable source evidence.` : '';
    latency.textContent = `${data.latency_ms.total} ms TOTAL · ${data.cache.hit ? 'CACHE HIT' : 'FRESH RETRIEVAL'}`;
    citations.replaceChildren(...data.retrieved_chunks.map((item, index) => {
      const card = document.createElement('article'); card.className = 'citation';
      const source = document.createElement('strong'); source.textContent = `[${index + 1}] ${item.citation.document} · p. ${item.citation.page}`;
      const excerpt = document.createElement('p'); excerpt.textContent = item.text;
      card.append(source, excerpt); return card;
    }));
    result.scrollIntoView({behavior:'smooth', block:'start'});
  } catch (error) { window.alert(error.message); }
  finally { askButton.disabled = false; askButton.querySelector('span').textContent = 'ASK THE ARCHIVE'; }
});
