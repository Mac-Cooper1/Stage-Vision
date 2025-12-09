/**
 * Stager Agent — Airtable → FastAPI trigger
 * Sends new order data to backend to start photo cleanup pipeline
 *
 * Form fields expected: Name, Email, Photos
 * (Address optional - will default to "order-{record_id}" if not provided)
 */

// Pull all inputs from Airtable automation config
const config = input.config();

// Photos: extract URLs and filenames from Airtable attachment objects
const rawPhotos = config.photos || [];
const photos = Array.isArray(rawPhotos)
  ? rawPhotos
      .filter(p => p && p.url)
      .map(p => ({
        url: p.url,
        filename: p.filename || "photo.jpg"
      }))
  : [];

// Build payload matching the Stager Agent webhook schema
const payload = {
  record_id: config.record_id || "",
  fields: {
    Name: config.name || "",
    Email: config.email || "",
    Address: config.address || `Order ${config.record_id || "unknown"}`,  // Default if no address field
    Occupied: "Yes",  // Default to occupied (declutter mode)
    Photos: photos
  }
};

// Your FastAPI endpoint - update this URL!
// For local dev: use ngrok URL
// For production: your deployed server
const webhookUrl = "YOUR_NGROK_URL/api/stager/airtable/webhook";

// Send the request
try {
  const response = await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  const result = await response.json();
  console.log("✅ Stager job started:", result);
} catch (err) {
  console.error("❌ Error:", String(err));
  throw err;  // Surfaces in Airtable Automation logs
}
