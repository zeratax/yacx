#!/usr/bin/env bash
_yacx() {
  COMPREPLY=()
  local word="${COMP_WORDS[COMP_CWORD]}"

  if [ "$COMP_CWORD" -eq 1 ]; then
  COMPREPLY=( $(compgen -W "execute-java execute-scala build-java build-scala" -- "$word") )
  elif [ "$COMP_CWORD" -eq 2 ]; then
    local words=("${COMP_WORDS[@]}")
    unset words[0]
    unset words[$COMP_CWORD]
    if [ "${words[@]}" == "execute-java" ]; then
      local completions=$(find examples/java -type f -iname "Example*.java" -exec basename '{}' \; | sed 's/\.java$//1')
      COMPREPLY=( $(compgen -W "$completions" -- "$word") )
    elif [ "${words[@]}" == "execute-scala" ]; then
      local completions=$(find examples/scala -type f -iname "Example*.scala" -exec basename '{}' \; | sed 's/\.scala$//1')
      COMPREPLY=( $(compgen -W "$completions" -- "$word") )
    fi
  fi
}

complete -F _yacx ./yacx.sh

