

export default function displayError(error) {
  if (error == null) return '';
  if (error instanceof Array) return error.join("\r\n");
  return error;
}
