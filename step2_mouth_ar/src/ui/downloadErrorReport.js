/** Snapshot immediately: edits or live frames may change while Excel loads. */
export async function downloadErrorReport(result, button) {
  if (!result || button.disabled) return;
  const snapshot = structuredClone(result), label = button.textContent;
  button.disabled = true; button.textContent = 'Preparing Excel…';
  try {
    const { createComparisonWorkbook } = await import('../eval/errorWorkbook.js');
    const book = await createComparisonWorkbook(snapshot);
    const bytes = await book.xlsx.writeBuffer();
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
    const link = document.createElement('a'); link.href = url;
    link.download = `dental-comparison-errors-${new Date().toISOString().slice(0, 10)}.xlsx`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);
    button.title = 'All error tables exported successfully';
  } catch (error) {
    button.title = `Excel export failed: ${error.message}`;
    button.textContent = 'Excel export failed — click to retry';
    return;
  } finally {
    button.disabled = false;
    if (button.textContent === 'Preparing Excel…') button.textContent = label;
  }
}
