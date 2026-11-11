// Recovery belongs to the application, not a repeated customer action.
// Unknown delivery outcomes always retain the same idempotency key.
const SEARCH_FAILURES = new Set(['destination_search_unavailable', 'quote_website_error', 'quote_error'])

export async function sendAdvisorRequest(request, { fetchImpl = globalThis.fetch,
  makeMessageId = () => `message-${globalThis.crypto.randomUUID()}`, timeoutMs = 45000, retryDelayMs = 600 } = {}) {
  let attemptRequest = request
  let failure
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt) await new Promise(resolve => setTimeout(resolve, retryDelayMs))
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const response = await fetchImpl('/advisor-chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(attemptRequest), signal: controller.signal,
      })
      const data = await response.json()
      if (!response.ok || data.status !== 'completed' || !data.reply) {
        const error = new Error('service_unavailable')
        error.customerReply = data.reply
        error.permanent = response.status >= 400 && response.status < 500 && ![408, 409, 429].includes(response.status)
        throw error
      }
      const readOnlyFailure = data.retryable && SEARCH_FAILURES.has(data.reason)
        && ['quote_request', 'hotel_list_request'].includes(data.intent)
        && data.tools_used?.includes('quote') && !data.offers?.length
      if (attempt === 0 && readOnlyFailure) {
        // This search completed and was cached; a new search needs a new key.
        attemptRequest = { ...request, message_id: makeMessageId() }
        continue
      }
      return data
    } catch (error) {
      failure = error
      if (error.permanent || attempt === 1) throw error
      // A disconnected request may already have created a lead. Recover its
      // stored response using the same key, never duplicate the operation.
    } finally {
      clearTimeout(timer)
    }
  }
  throw failure || new Error('service_unavailable')
}
