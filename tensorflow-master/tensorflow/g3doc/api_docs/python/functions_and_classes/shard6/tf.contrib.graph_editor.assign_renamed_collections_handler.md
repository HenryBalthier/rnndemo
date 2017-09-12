### `tf.contrib.graph_editor.assign_renamed_collections_handler(info, elem, elem_)` {#assign_renamed_collections_handler}

Add the transformed elem to the (renamed) collections of elem.

A collection is renamed only if is not a known key, as described in
`tf.GraphKeys`.

##### Args:


*  <b>`info`</b>: Transform._TmpInfo instance.
*  <b>`elem`</b>: the original element (`tf.Tensor` or `tf.Operation`)
*  <b>`elem_`</b>: the transformed element

