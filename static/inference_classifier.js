// static/inference_classifier.js

;(function(window){
  /**
   * Send the base64-encoded image to the server for inference.
   * @param {string} base64img - Data URL like "data:image/jpeg;base64,...."
   * @returns {Promise<string>} - Resolves to the predicted label.
   */
  async function predict(base64img) {
    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: base64img })
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      return data.label;
    } catch (err) {
      console.error('Inference error:', err);
      return '';
    }
  }

  // Expose on window so your HTML can just call inference_classifier.predict(...)
  window.inference_classifier = {
    predict
  };
})(window);
