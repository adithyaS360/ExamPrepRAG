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
      question: input.value
    }),
    signal: controller.signal
  });

  // existing response handling...

} catch (error) {
  if (error.name === 'AbortError') {
    throw new Error(
      'The archive took too long to respond. Check the Render logs.'
    );
  }

  throw error;

} finally {
  clearTimeout(timeout);
}