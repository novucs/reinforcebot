

export function displayErrors(...errors) {
  return errors.flatMap((e) => displayError(e));
}

export default function displayError(error) {
  if (error == null) return [];
  if (error instanceof Array) return error;
  return [error];
}
