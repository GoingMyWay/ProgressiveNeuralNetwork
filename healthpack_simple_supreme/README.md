Description
---

To train it, you should first train an agent on healpack gathering simple scenario, you can use the code of [ViZDoomAgents](https://github.com/GoingMyWay/ViZDoomAgents/tree/master/healthpack_gathering) to a simple health pack gethering model, Note that do not forget to add scope name to the scope of network as you can see in the code below

    def __create_network(self, scope, img_shape=(80, 80)):
        with tf.variable_scope(TASK_NAME_HERE):  # to insert a new scope here
            with tf.variable_scope(scope):
                ...


