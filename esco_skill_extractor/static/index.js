function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function normalizeItems(entries) {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.map((s) => {
    if (typeof s === "string") {
      return { href: s, text: s, reason: "" };
    }
    const uri = s && s.uri;
    const label = s && (s.label_en ?? s.labelEn ?? s.label_it ?? s.labelIt);
    const reason = (s && s.reason) || "";
    return {
      href: uri || "#",
      text: label || uri || "",
      reason,
    };
  });
}

function renderList(items) {
  if (!items.length) {
    return "<p class=\"empty-section\">None found.</p>";
  }
  return `<ul>
    ${items
      .map((item) => {
        const reasonBlock = item.reason
          ? `<div class="skill-reason">${escapeHtml(item.reason)}</div>`
          : "";
        return `<li>
            <a href="${escapeHtml(item.href)}">${escapeHtml(item.text)}</a>
            ${reasonBlock}
          </li>`;
      })
      .join("")}
  </ul>`;
}

async function extractEntity(event) {
  event.preventDefault();

  const text = document.getElementById("text").value;
  const output = document.getElementById("output");
  const submitButton = document.getElementById("submit-button");

  submitButton.disabled = true;
  submitButton.textContent = "Extracting…";
  output.innerHTML = "<p>Running extraction…</p>";

  try {
    const response = await fetch(`${window.SERVER}/extract`, {
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

    const row =
      Array.isArray(data) && data[0] && typeof data[0] === "object"
        ? data[0]
        : { skills: [], occupations: [] };

    const skillItems = normalizeItems(row.skills);
    const occupationItems = normalizeItems(row.occupations);
    const copyText = [
      ...skillItems.map((i) => i.text),
      ...occupationItems.map((i) => i.text),
    ].join(", ");

    output.innerHTML = `
      <button type="button" id="copyButton">Copy CSV</button>
      <section class="result-section">
        <h2>Skills</h2>
        ${renderList(skillItems)}
      </section>
      <section class="result-section">
        <h2>Occupations</h2>
        ${renderList(occupationItems)}
      </section>
    `;
    document.getElementById("copyButton").addEventListener("click", () => {
      copyToClipboard(copyText);
    });
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Extract skills and occupations";
  }
}

async function copyToClipboard(text) {
  try {
    const button = document.getElementById("copyButton");
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied!";
    setTimeout(() => (button.textContent = "Copy CSV"), 1000);
  } catch (err) {
    console.error("Failed to copy: ", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) extractEntity(event);
  });
});
