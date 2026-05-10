function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * This function is called when the form is submitted to ask the server to extract an entity from the text.
 * Then it updates the DOM with the extracted entities.
 * @param {Event} event the event object of the form submission.
 */
async function extractEntity(event) {
  event.preventDefault();

  const entity = document.querySelector('input[name="entity"]:checked').value;
  const text = document.getElementById("text").value;
  const output = document.getElementById("output");

  const response = await fetch(`${window.SERVER}/extract-${entity}`, {
    method: "POST",
    body: JSON.stringify([text]),
    headers: {
      "Content-Type": "application/json",
    },
  });

  const data = await response.json();
  if (!response.ok) {
    const msg =
      data && typeof data.error === "string" ? data.error : "Request failed";
    output.innerHTML = `<p>${escapeHtml(msg)}</p>`;
    return;
  }

  const row = Array.isArray(data) && Array.isArray(data[0]) ? data[0] : [];
  const items =
    entity === "skills"
      ? row.map((s) => {
          if (typeof s === "string") {
            return { href: s, text: s, reason: "" };
          }
          const uri = s && s.uri;
          const label =
            s && (s.label_en ?? s.labelEn ?? s.label_it ?? s.labelIt);
          const reason = (s && s.reason) || "";
          return {
            href: uri || "#",
            text: label || uri || "",
            reason,
          };
        })
      : row.map((u) => ({ href: u, text: u, reason: "" }));
  const skillsStr = items.map((i) => i.text).join(", ");

  output.innerHTML = `
    <button type="button" id="copyButton"> Copy CSV </button>
    <ul>
      ${items
        .map((item) => {
          const reasonBlock = item.reason
            ? `<div class="skill-reason">${escapeHtml(item.reason)}</div>`
            : "";
          return `<li>
              <a href="${escapeHtml(item.href)}"> ${escapeHtml(item.text)} </a>
              ${reasonBlock}
            </li>`;
        })
        .join("")}
    </ul>`;
  document.getElementById("copyButton").addEventListener("click", () => {
    copyToClipboard(skillsStr);
  });
}

/**
 * This function is used to copy some text to the clipboard.
 * @param {string} text the text to be copied to the clipboard.
 */
async function copyToClipboard(text) {
  try {
    const button = document.getElementById("copyButton");
    await navigator.clipboard.writeText(text);
    button.innerHTML = "Copied!";
    setTimeout(() => (button.innerHTML = "Copy CSV"), 1000);
  } catch (err) {
    console.error("Failed to copy: ", err);
  }
}

/**
 * This function renders the texts based on the selected entity.
 */
function renderTexts() {
  const entity = document.querySelector('input[name="entity"]:checked').value;
  const textarea = document.getElementById("text");
  const submitButton = document.getElementById("submit-button");
  const output = document.getElementById("output");

  textarea.setAttribute(
    "placeholder",
    `Paste your text here to extract ${entity}`
  );

  submitButton.innerHTML = `Extract ${entity}`;

  output.innerHTML = `Extracted ${entity} will appear here`;
}

document.addEventListener("DOMContentLoaded", () => {
  // Render texts based on the selected entity.
  renderTexts();

  // Update the texts when the entity is changed.
  document
    .querySelectorAll('input[name="entity"]')
    .forEach((radio) => radio.addEventListener("change", renderTexts));

  // Submit the form with ctrl+enter when the textarea is focused.
  document.getElementById("text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) extractEntity(event);
  });
});
