import React, {Component} from "react";
import {Button, Container, Menu} from "semantic-ui-react";

export default class TopMenu extends Component {
  render() {
    return (
      <Menu fixed='top' size='large'>
        <Container>
          <Menu.Item as='a' href='/'>
            Home
          </Menu.Item>
          <Menu.Item position='right'>
            <Button as='a' href='/signin'>
              Sign in
            </Button>
            <Button as='a' href='/signup' primary style={{marginLeft: '0.5em'}}>
              Sign Up
            </Button>
          </Menu.Item>
        </Container>
      </Menu>
    )
  }
}
